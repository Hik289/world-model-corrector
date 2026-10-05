from __future__ import annotations
import os
import time
import json
import re
from dataclasses import dataclass
from typing import Optional


try:
    import openai as _openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    import google.genai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


@dataclass
class LLMResult:
    identified_steps: list[int]
    repair_summary: str
    confidence: float
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw_response: str
    usage_estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


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


class LLMClient:


    SUPPORTED = {

        "gpt-3.5-turbo":        "openai",
        "gpt-4o-mini":          "openai",
        "gpt-4o":               "openai",
        "gpt-4.1-nano":         "openai",
        "gpt-4.1-mini":         "openai",
        "gpt-4.1":              "openai",
        "gpt-4-turbo":          "openai",

        "gemini-2.5-flash":     "gemini",
        "gemini-2.5-flash-lite":"gemini",
        "gemini-2.5-pro":       "gemini",

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

        self.backend = (
            backend
            or os.environ.get("LLM_BACKEND")
            or self.SUPPORTED.get(self.model, "openai")
        ).lower()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")


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
            if not key:
                raise ValueError(
                    "missing LLM_API_KEY or OPENAI_API_KEY for OpenAI backend"
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
            if not key:
                raise ValueError(
                    "missing LLM_API_KEY or GEMINI_API_KEY for Gemini backend"
                )
            self._gemini_client = genai.Client(api_key=key)
            self._gemini_model = self.model

        else:
            raise ValueError(f"Unsupported LLM backend: {self.backend}")


    def _call(self, system: str, user: str) -> tuple[str, Optional[int], Optional[int], float]:

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
                    usage = getattr(resp, "usage", None)
                    pt = getattr(usage, "prompt_tokens", None)
                    ct = getattr(usage, "completion_tokens", None)

                elif self.backend == "gemini":
                    prompt = f"{system}\n\n{user}"
                    resp = self._gemini_client.models.generate_content(
                        model=self._gemini_model,
                        contents=prompt,
                    )
                    raw = resp.text or ""

                    usage = getattr(resp, "usage_metadata", None)
                    pt = getattr(usage, "prompt_token_count", None)
                    ct = getattr(usage, "candidates_token_count", None)

                pt = pt if isinstance(pt, int) and not isinstance(pt, bool) and pt >= 0 else None
                ct = ct if isinstance(ct, int) and not isinstance(ct, bool) and ct >= 0 else None
                latency = (time.time() - t0) * 1000
                return raw, pt, ct, latency

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM call failed after {self.max_retries} tries: {e}") from e

        raise RuntimeError("unreachable")


    @staticmethod
    def _legacy_usage(pt, ct, system, user, raw):
        estimated = pt is None or ct is None
        if pt is None:
            pt = len(f"{system}\n\n{user}".split()) * 4 // 3
        if ct is None:
            ct = len(raw.split()) * 4 // 3
        return pt, ct, estimated


    @staticmethod
    def _parse_json(raw: str) -> dict:


        text = re.sub(r"```(?:json)?\n?", "", raw).strip()

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
        try:
            return json.loads(text)
        except json.JSONDecodeError:

            return {}


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


    def locate_error(
        self,
        steps: list[dict],
        failure_desc: str = "task failed",
    ) -> LLMResult:

        steps_text = self._format_steps(steps)
        user = _LOCATE_USER.format(
            n_steps=len(steps),
            failure_desc=failure_desc,
            steps_text=steps_text,
        )
        raw, pt, ct, lat = self._call(_LOCATE_SYSTEM, user)
        pt, ct, estimated = self._legacy_usage(pt, ct, _LOCATE_SYSTEM, user, raw)
        parsed = self._parse_json(raw)
        return LLMResult(
            identified_steps=parsed.get("root_cause_steps", []),
            repair_summary=parsed.get("fix", parsed.get("issue", "")),
            confidence=float(parsed.get("confidence", 0.5)),
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=lat,
            raw_response=raw,
            usage_estimated=estimated,
        )

    def repair_region(
        self,
        region_steps: list[dict],
        failure_desc: str = "task failed",
    ) -> LLMResult:

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
        pt, ct, estimated = self._legacy_usage(pt, ct, _REPAIR_SYSTEM, user, raw)
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
            usage_estimated=estimated,
        )

    def full_replan(
        self,
        all_steps: list[dict],
        failure_desc: str = "task failed",
    ) -> LLMResult:

        steps_text = self._format_steps(all_steps)
        user = _FULLPLAN_USER.format(
            n_steps=len(all_steps),
            failure_desc=failure_desc,
            steps_text=steps_text,
        )
        raw, pt, ct, lat = self._call(_FULLPLAN_SYSTEM, user)
        pt, ct, estimated = self._legacy_usage(pt, ct, _FULLPLAN_SYSTEM, user, raw)
        parsed = self._parse_json(raw)
        return LLMResult(
            identified_steps=parsed.get("root_cause_steps", []),
            repair_summary=parsed.get("explanation", ""),
            confidence=float(parsed.get("confidence", 0.5)),
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=lat,
            raw_response=raw,
            usage_estimated=estimated,
        )

    def chat(self, system: str, user: str) -> "ChatResult":

        raw, pt, ct, lat = self._call(system, user)
        return ChatResult(
            text=raw,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=lat,
        )


@dataclass
class ChatResult:
    text: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency_ms: float

    @property
    def total_tokens(self) -> Optional[int]:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens
