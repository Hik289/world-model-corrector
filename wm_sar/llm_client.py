"""
llm_client.py — Model API calls for WM-SAR experiments.

The default path reads general model settings from environment variables:
  LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_BACKEND

Provider-specific arguments and environment variables are still accepted for
backward compatibility with older experiment scripts.

Usage:
  client = LLMClient()
  result = client.locate_error(trace_steps)
  result = client.repair_region(region_steps)
  result = client.full_replan(trace_steps)
"""

from __future__ import annotations
import os
import time
import json
import re
from dataclasses import dataclass, field
from typing import Optional

# ── external deps ──────────────────────────────────────────────────────────
try:
    import openai as _openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    import google.genai as genai
    from google.genai import types as genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai  # fallback to old SDK
        _GEMINI_AVAILABLE = True
    except ImportError:
        _GEMINI_AVAILABLE = False

# ── result dataclass ────────────────────────────────────────────────────────
@dataclass
class LLMResult:
    identified_steps: list[int]   # steps the LLM identified as root cause
    repair_summary: str            # one-sentence description of the fix
    confidence: float              # 0-1 self-reported confidence
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw_response: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

# ── prompt templates ────────────────────────────────────────────────────────
_LOCATE_SYSTEM = (
    "You are an expert agent-failure analyst. "
    "Given a window of an agent's world-model rollout, "
    "identify the step(s) most likely to contain the root cause of the failure. "
    "Be concise. Respond in JSON only."
)

_LOCATE_USER = """\
The following is a window of {n_steps} consecutive steps from a failed world-model rollout.
Each step shows what the agent PREDICTED would happen vs what ACTUALLY happened.
The final failure is: {failure_desc}

STEPS:
{steps_text}

Identify the root-cause step(s). Respond ONLY with valid JSON:
{{
  "root_cause_steps": [<list of step numbers, 1-indexed>],
  "issue": "<one sentence describing the error>",
  "fix": "<one sentence describing the repair>",
  "confidence": <0.0-1.0>
}}"""

_REPAIR_SYSTEM = (
    "You are an expert agent repair system. "
    "You receive a connected subgraph region identified by graph error amplification analysis "
    "as the likely source of cascading failures. "
    "Repair this region as a unit by correcting ALL steps together. "
    "Respond in JSON only."
)

_REPAIR_USER = """\
The following REGION of {n_steps} steps has been identified by spectral graph analysis as the
error-amplifying subgraph. These steps must be repaired as a unit to restore consistency.

REGION STEPS (steps {min_step}-{max_step}):
{steps_text}

FAILURE TARGET: {failure_desc}

Provide a coherent repair for all steps in this region. Respond ONLY with valid JSON:
{{
  "repaired_steps": [<same step numbers as input>],
  "repairs": {{
    "<step_number>": "<corrected predicted state in one sentence>"
  }},
  "explanation": "<one paragraph explaining the root cause and how the repair restores consistency>",
  "confidence": <0.0-1.0>
}}"""

_FULLPLAN_SYSTEM = (
    "You are an expert agent-failure analyst with access to the complete rollout. "
    "Identify the root cause and provide a comprehensive repair plan. "
    "Respond in JSON only."
)

_FULLPLAN_USER = """\
The following is the COMPLETE failed world-model rollout ({n_steps} steps total).
The final failure is: {failure_desc}

COMPLETE ROLLOUT:
{steps_text}

Identify the root cause step(s) and provide a repair plan. Respond ONLY with valid JSON:
{{
  "root_cause_steps": [<list of step numbers, 1-indexed>],
  "repair_plan": {{
    "<step_number>": "<corrected predicted state>"
  }},
  "explanation": "<one paragraph>",
  "confidence": <0.0-1.0>
}}"""


# ── LLM client ──────────────────────────────────────────────────────────────
class LLMClient:
    """Unified LLM client with general env-var configuration."""

    SUPPORTED = {
        # OpenAI models
        "gpt-3.5-turbo":        "openai",
        "gpt-4o-mini":          "openai",
        "gpt-4o":               "openai",
        "gpt-4.1-nano":         "openai",
        "gpt-4.1-mini":         "openai",
        "gpt-4.1":              "openai",
        "gpt-4-turbo":          "openai",
        # Google Gemini models
        "gemini-2.5-flash":     "gemini",
        "gemini-2.5-flash-lite":"gemini",
        "gemini-2.5-pro":       "gemini",
        # legacy aliases kept for API compatibility (will 404 at call time, see note)
        "gemini-2.0-flash":     "gemini",
        "gemini-2.0-flash-lite":"gemini",
        "gemini-1.5-flash":     "gemini",
        "gemini-1.5-pro":       "gemini",
        "gemini-flash":         "gemini",
    }

    def __init__(
        self,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.model = (
            model
            or os.environ.get("LLM_MODEL")
            or os.environ.get("MODEL_NAME")
            or "gpt-4o-mini"
        )
        # backend can be explicitly set, read from env, or inferred from known aliases
        self.backend = (
            backend
            or os.environ.get("LLM_BACKEND")
            or self.SUPPORTED.get(self.model, "openai")
        ).lower()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens

        # initialise backend
        if self.backend in {
            "openai",
            "openai-compatible",
            "chat",
            "chat-completions",
            "chat_completions",
            "compatible",
        }:
            self.backend = "openai"
            if not _OPENAI_AVAILABLE:
                raise ImportError("pip install openai")
            key = (
                api_key
                or openai_api_key
                or os.environ.get("LLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY", "")
            )
            endpoint = (
                base_url
                or os.environ.get("LLM_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
            )
            client_kwargs = {"api_key": key}
            if endpoint:
                client_kwargs["base_url"] = endpoint
            self._openai = _openai.OpenAI(**client_kwargs)

        elif self.backend == "gemini":
            if not _GEMINI_AVAILABLE:
                raise ImportError("pip install google-genai")
            key = (
                api_key
                or gemini_api_key
                or os.environ.get("LLM_API_KEY")
                or os.environ.get("GEMINI_API_KEY", "")
            )
            self._gemini_client = genai.Client(api_key=key)
            self._gemini_model = self.model

        else:
            raise ValueError(f"Unsupported LLM backend: {self.backend}")

    # ── internal call ────────────────────────────────────────────────────────
    def _call(self, system: str, user: str) -> tuple[str, int, int, float]:
        """Returns (raw_text, prompt_tokens, completion_tokens, latency_ms)."""
        t0 = time.time()
        for attempt in range(self.max_retries):
            try:
                if self.backend == "openai":
                    resp = self._openai.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    raw = resp.choices[0].message.content or ""
                    pt  = resp.usage.prompt_tokens
                    ct  = resp.usage.completion_tokens

                elif self.backend == "gemini":
                    prompt = f"{system}\n\n{user}"
                    resp = self._gemini_client.models.generate_content(
                        model=self._gemini_model,
                        contents=prompt,
                    )
                    raw = resp.text or ""
                    # approximate token counts from usage metadata if available
                    usage = getattr(resp, "usage_metadata", None)
                    pt = getattr(usage, "prompt_token_count", len(prompt.split()) * 4 // 3)
                    ct = getattr(usage, "candidates_token_count", len(raw.split()) * 4 // 3)

                latency = (time.time() - t0) * 1000
                return raw, pt, ct, latency

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM call failed after {self.max_retries} tries: {e}") from e

        raise RuntimeError("unreachable")

    # ── parse JSON from LLM output ───────────────────────────────────────────
    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract JSON from raw LLM output (handles markdown fences)."""
        # strip markdown code fences
        text = re.sub(r"```(?:json)?\n?", "", raw).strip()
        # find the first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # fallback: return empty
            return {}

    # ── format steps for LLM ────────────────────────────────────────────────
    @staticmethod
    def _format_steps(steps: list[dict]) -> str:
        lines = []
        for s in steps:
            t = s.get("step", "?")
            pred = s.get("predicted", "—")
            actual = s.get("actual", "—")
            err = s.get("error", 0.0)
            lines.append(
                f"  Step {t}: PREDICTED={pred!r}  |  ACTUAL={actual!r}  |  error={err:.3f}"
            )
        return "\n".join(lines)

    # ── public API ───────────────────────────────────────────────────────────
    def locate_error(
        self,
        steps: list[dict],
        failure_desc: str = "task failed",
    ) -> LLMResult:
        """Ask LLM to locate the root-cause step(s) in a window of steps."""
        steps_text = self._format_steps(steps)
        user = _LOCATE_USER.format(
            n_steps=len(steps),
            failure_desc=failure_desc,
            steps_text=steps_text,
        )
        raw, pt, ct, lat = self._call(_LOCATE_SYSTEM, user)
        parsed = self._parse_json(raw)
        return LLMResult(
            identified_steps=parsed.get("root_cause_steps", []),
            repair_summary=parsed.get("fix", parsed.get("issue", "")),
            confidence=float(parsed.get("confidence", 0.5)),
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=lat,
            raw_response=raw,
        )

    def repair_region(
        self,
        region_steps: list[dict],
        failure_desc: str = "task failed",
    ) -> LLMResult:
        """Ask LLM to repair a connected subgraph region as a unit."""
        if not region_steps:
            return LLMResult([], "", 0.5, 0, 0, 0.0, "")
        step_nums = [s.get("step", 0) for s in region_steps]
        steps_text = self._format_steps(region_steps)
        user = _REPAIR_USER.format(
            n_steps=len(region_steps),
            min_step=min(step_nums),
            max_step=max(step_nums),
            steps_text=steps_text,
            failure_desc=failure_desc,
        )
        raw, pt, ct, lat = self._call(_REPAIR_SYSTEM, user)
        parsed = self._parse_json(raw)
        repaired = list(parsed.get("repaired_steps", step_nums))
        return LLMResult(
            identified_steps=repaired,
            repair_summary=parsed.get("explanation", ""),
            confidence=float(parsed.get("confidence", 0.5)),
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=lat,
            raw_response=raw,
        )

    def full_replan(
        self,
        all_steps: list[dict],
        failure_desc: str = "task failed",
    ) -> LLMResult:
        """Ask LLM to locate root cause from the full trace."""
        steps_text = self._format_steps(all_steps)
        user = _FULLPLAN_USER.format(
            n_steps=len(all_steps),
            failure_desc=failure_desc,
            steps_text=steps_text,
        )
        raw, pt, ct, lat = self._call(_FULLPLAN_SYSTEM, user)
        parsed = self._parse_json(raw)
        return LLMResult(
            identified_steps=parsed.get("root_cause_steps", []),
            repair_summary=parsed.get("explanation", ""),
            confidence=float(parsed.get("confidence", 0.5)),
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=lat,
            raw_response=raw,
        )


# ── Generic chat interface ───────────────────────────────────────────────────

from dataclasses import dataclass as _dataclass

@_dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def chat(self, system: str, user: str) -> ChatResult:
    """Generic chat call. Returns ChatResult with .text, .prompt_tokens, etc."""
    raw, pt, ct, lat = self._call(system, user)
    return ChatResult(text=raw, prompt_tokens=pt, completion_tokens=ct, latency_ms=lat)


# Monkey-patch onto LLMClient
LLMClient.chat = chat
