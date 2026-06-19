"""
text_scenarios.py — Converts numeric synthetic rollouts into textual
descriptions that LLMs can reason about.

Each rollout is mapped to a realistic task scenario (booking, code review,
data pipeline, etc.). Each step gets:
  - predicted_text: what the world model imagined
  - actual_text:   what actually happened
  - error_type:    high-level category

This bridges the gap between numeric simulation and LLM-based repair.
"""

from __future__ import annotations
import numpy as np
from typing import Any

# ── Scenario templates ──────────────────────────────────────────────────────
# Each scenario provides step templates: (predicted_ok, actual_ok, error_variant)
# {field} placeholders are filled with step-specific values.

SCENARIOS = {
    "flight_booking": {
        "task": "Book a round-trip flight from {city_a} to {city_b} for 2 passengers, economy class, departing {date}.",
        "failure_desc": "Booking failed: confirmation never received, credit card charged but no tickets issued.",
        "step_templates": [
            ("Search flights for {city_a}-{city_b} on {date}: found 12 options, cheapest $340.",
             "Search returned 12 results but prices ranged $340-$890.",
             "Search result mis-cached: predicted cheapest was $340 but cache returned stale $290 from 3 days ago."),
            ("Select flight UA-{num}: departs 08:15, arrives 14:30, 2 seats available.",
             "Flight UA-{num} selected: 08:15 departure confirmed.",
             "Seat availability check failed silently: predicted 2 seats available but inventory showed 0 in economy."),
            ("Fill passenger details: seat 14A and 14B assigned, meal preference noted.",
             "Passenger details accepted: reference code PAX-{num}.",
             "Seat assignment API timed out: predicted seats 14A/14B but system auto-assigned 32F/32G (back row)."),
            ("Submit payment: Visa ending 4242, total $680 for 2 tickets.",
             "Payment processed: transaction ID TXN-{num}.",
             "Payment gateway returned PENDING instead of CONFIRMED: predicted CONFIRMED status, actual PENDING."),
            ("Receive e-ticket: confirmation code BKG-{num} emailed to user.",
             "Confirmation email sent to user@email.com.",
             "Confirmation step skipped due to PENDING payment: no email sent, user has no tickets despite charge."),
            ("Booking complete: PNR {num}, seats 14A/14B, meal preferences saved.",
             "Booking record created in system.",
             "PNR creation failed: booking record marked INCOMPLETE, seats not held."),
        ],
    },
    "code_review": {
        "task": "Review PR #{num} (adding user authentication module), verify tests pass, check security.",
        "failure_desc": "Code merged with critical SQL injection vulnerability in login function.",
        "step_templates": [
            ("Fetch PR #{num}: 847 lines changed, 23 files, 3 reviewers assigned.",
             "PR fetched: 847 lines changed across 23 files.",
             "PR metadata fetch returned wrong branch: predicted main←feature/auth, actual fetched stale draft branch."),
            ("Run CI pipeline: all 142 tests pass, coverage 87%.",
             "CI pipeline triggered, results pending.",
             "CI result mis-read: predicted all tests pass (142/142) but actual showed 3 failures in auth_test.py."),
            ("Check security: no SQL injection, passwords hashed with bcrypt.",
             "Static analysis ran: reported 0 critical issues.",
             "Security scanner skipped login.py due to config error: predicted clean scan, actual login.py unchecked."),
            ("Review DB queries: all parameterized, ORM used correctly.",
             "DB query review: 8 queries inspected.",
             "login() function uses raw f-string SQL: predicted parameterized, actual f\"SELECT * WHERE user='{input}'\"."),
            ("Approve PR: all checks green, security confirmed, approving.",
             "Approval submitted by reviewer.",
             "Approval granted despite unresolved security issue: predicted all checks green, actual 1 critical unfixed."),
            ("Merge PR #{num} into main: squash-merge, CI reruns.",
             "PR merged into main branch.",
             "Vulnerability merged to production: SQL injection in login endpoint now live."),
        ],
    },
    "data_pipeline": {
        "task": "Run nightly ETL pipeline: ingest sales data, transform, load to warehouse, send report.",
        "failure_desc": "Daily report sent with yesterday's stale data; warehouse not updated.",
        "step_templates": [
            ("Connect to source DB: read 48,231 new sales records since last run.",
             "Source DB connection established.",
             "Source DB connection used wrong cursor offset: predicted 48,231 new records, actual re-read 0 (offset bug)."),
            ("Validate schema: all 18 required fields present, no nulls in key columns.",
             "Schema validation passed for 48,231 rows.",
             "Null check skipped for 'transaction_id' column: predicted 0 nulls, actual 1,847 nulls in loaded data."),
            ("Transform: apply exchange rates, normalize currencies to USD.",
             "Currency transformation applied to all records.",
             "Exchange rate cache stale (48h old): predicted current EUR/USD=1.08, actual applied 1.12 from Tuesday."),
            ("Load to warehouse: UPSERT 48,231 rows to sales_fact table.",
             "Warehouse load completed.",
             "UPSERT failed silently due to schema lock: predicted 48,231 rows loaded, actual 0 rows inserted."),
            ("Verify load: row count matches source, checksums equal.",
             "Verification query returned match.",
             "Verification query counted stale rows from prior day: predicted match confirmed, actual mismatch undetected."),
            ("Generate and send daily report: revenue $2.3M, 48,231 transactions.",
             "Report emailed to 12 stakeholders.",
             "Report generated from stale warehouse (prior day data): sent incorrect figures, stakeholders misled."),
        ],
    },
    "api_orchestration": {
        "task": "Orchestrate multi-step API workflow: fetch user profile, check subscription, deliver content.",
        "failure_desc": "Premium content delivered to free-tier user; subscription check bypassed.",
        "step_templates": [
            ("Fetch user profile for user_id={num}: name, email, account tier.",
             "User profile fetched: user_id={num}, tier=unknown.",
             "Profile API returned cached stale response: predicted fresh tier=premium, actual stale tier=free (3h old)."),
            ("Authenticate JWT token: valid, not expired, scope=read:content.",
             "JWT validated: signature OK, expiry OK.",
             "Scope claim mis-parsed: predicted scope includes 'read:content', actual scope='read:profile' only."),
            ("Check subscription: user has active premium subscription, expires 2025-12-31.",
             "Subscription service queried.",
             "Subscription service timeout: predicted active premium, actual TIMEOUT treated as 'pass' (fail-open bug)."),
            ("Authorize content access: subscription valid, user allowed premium tier.",
             "Authorization decision: ALLOW.",
             "Authorization used stale cached decision: predicted re-check subscription, actual used 2h cached ALLOW."),
            ("Deliver premium content: article ID={num}, full text, no ads.",
             "Content delivered to user.",
             "Free-tier user received premium content: subscription check was bypassed due to timeout fail-open."),
            ("Log access event: user_id={num}, content_id={num}, tier=premium.",
             "Access log written.",
             "Log recorded incorrect tier: actual free-tier user logged as premium, audit trail corrupted."),
        ],
    },
    "research_agent": {
        "task": "Research task: find top-5 papers on '{topic}', summarize findings, draft literature review section.",
        "failure_desc": "Literature review contains fabricated citations that don't exist.",
        "step_templates": [
            ("Search Semantic Scholar for '{topic}': query returns 234 papers, filter top-cited.",
             "Search executed: 234 results for '{topic}'.",
             "Search API paginated incorrectly: predicted top-cited papers from 234 results, actual returned page 3 (random)."),
            ("Retrieve paper details: titles, authors, abstracts, DOIs for top 10 candidates.",
             "Paper metadata fetched for 10 papers.",
             "DOI resolution failed for 3 papers: predicted valid DOIs, actual 3 DOIs returned 404 (broken links)."),
            ("Verify paper existence: confirm all 10 papers accessible via DOI.",
             "Verification: 10/10 papers accessible.",
             "Verification skipped due to rate limit: predicted all papers verified, actual 3 unverified papers included."),
            ("Select top 5 papers by citation count and relevance.",
             "Top 5 papers selected.",
             "Citation counts from cache (stale 30d): predicted correct citation ranking, actual outdated counts used."),
            ("Summarize selected papers: extract key findings, methods, results.",
             "Summaries generated for 5 papers.",
             "LLM hallucinated summary for one unverified paper: predicted accurate summary, actual fabricated content."),
            ("Draft literature review: weave summaries into coherent narrative with citations.",
             "Literature review draft completed.",
             "Draft contains fabricated citation: [Author et al., 2024] does not exist, was hallucinated in step 5."),
        ],
    },
}

CITIES = ["NYC", "LAX", "LHR", "CDG", "NRT", "SYD", "DXB", "SIN"]
DATES = ["2025-08-15", "2025-09-02", "2025-10-20", "2025-11-05"]
TOPICS = [
    "graph neural networks for temporal forecasting",
    "LLM agent world models",
    "multi-agent cooperation under partial observability",
    "tool-augmented language model planning",
]


def rollout_to_steps(
    rollout: Any,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """
    Convert a numeric AgentRollout to a list of text-annotated step dicts.

    Each dict has:
      step (int), predicted (str), actual (str), error (float),
      uncertainty (float), error_type (str), is_root_cause (bool)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # pick a scenario
    scenario_keys = list(SCENARIOS.keys())
    scenario_name = scenario_keys[int(rng.integers(0, len(scenario_keys)))]
    scenario = SCENARIOS[scenario_name]

    # fill in placeholders
    num = int(rng.integers(1000, 9999))
    city_a = CITIES[int(rng.integers(0, 4))]
    city_b = CITIES[int(rng.integers(4, 8))]
    date = DATES[int(rng.integers(0, len(DATES)))]
    topic = TOPICS[int(rng.integers(0, len(TOPICS)))]

    def fill(s: str) -> str:
        return (s
                .replace("{num}", str(num))
                .replace("{city_a}", city_a)
                .replace("{city_b}", city_b)
                .replace("{date}", date)
                .replace("{topic}", topic))

    task_desc = fill(scenario["task"])
    failure_desc = fill(scenario["failure_desc"])
    templates = scenario["step_templates"]

    steps = []
    agent_steps = getattr(rollout, "steps", [])
    root_t = getattr(rollout, "root_cause_t", -1)

    for i, agent_step in enumerate(agent_steps):
        t = agent_step.t
        tmpl_idx = i % len(templates)
        pred_ok, actual_ok, error_variant = templates[tmpl_idx]

        is_root = (t == root_t)
        is_downstream = getattr(agent_step, "downstream_of_error", False)
        numeric_err = agent_step.prediction_error

        if is_root:
            predicted = fill(pred_ok)
            actual = fill(error_variant)          # ← error injected here
            error_type = "root_cause"
        elif is_downstream and numeric_err > 0.15:
            # downstream corruption: predicted was ok but propagated error
            predicted = fill(pred_ok)
            actual = fill(actual_ok) + f" [propagated mismatch: downstream of step {root_t}]"
            error_type = "downstream_corruption"
        else:
            predicted = fill(pred_ok)
            actual = fill(actual_ok)
            error_type = "ok"

        steps.append({
            "step": t,
            "predicted": predicted,
            "actual": actual,
            "error": float(numeric_err),
            "uncertainty": float(agent_step.uncertainty),
            "error_type": error_type,
            "is_root_cause": is_root,
            "status": agent_step.status,
        })

    return steps, task_desc, failure_desc
