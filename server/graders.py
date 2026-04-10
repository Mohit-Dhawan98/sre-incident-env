"""Standalone grader functions for OpenEnv task evaluation.

Referenced from openenv.yaml tasks block. Each function grades an episode
for a specific difficulty tier using the shared compute_reward() function.

The graders accept the agent's response and scenario context, run the
evaluation through our 7-component reward function, and return a score
in [0.01, 0.99].

These functions can be imported and called directly by the OpenEnv validator:
    from server.graders import grade_easy, grade_medium, grade_hard
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from server.reward import compute_reward


def _grade_episode(
    submitted_root_cause: str = "",
    submitted_service: str = "",
    submitted_failure_type: str = "",
    true_root_service: str = "",
    true_failure_type: str = "",
    services_graph: Optional[Dict[str, Any]] = None,
    explanation_keywords: Optional[List[str]] = None,
    system_healthy: bool = False,
    harm_count: int = 0,
    optimal_steps: int = 0,
    remediation_count: int = 0,
    observation_after_fix: int = 0,
    discovered_before_action: int = 0,
    execute_runbook_count: int = 0,
    unique_remediation_count: int = 0,
    progress_state_visits: int = 0,
) -> float:
    """Internal: compute reward for an episode with explicit parameters."""
    return compute_reward(
        submitted_root_cause=submitted_root_cause,
        submitted_service=submitted_service,
        true_root_service=true_root_service,
        submitted_failure_type=submitted_failure_type,
        true_failure_type=true_failure_type,
        services_graph=services_graph or {},
        explanation_keywords=explanation_keywords or [],
        system_healthy=system_healthy,
        harm_count=harm_count,
        optimal_steps=optimal_steps,
        remediation_count=remediation_count,
        observation_after_fix=observation_after_fix,
        discovered_before_action=discovered_before_action,
        execute_runbook_count=execute_runbook_count,
        unique_remediation_count=unique_remediation_count,
        progress_state_visits=progress_state_visits,
    )


def grade_easy(response: str = "", scenario: Optional[Dict] = None, **kwargs) -> float:
    """Grade an easy-tier episode.

    For single-response evaluation (validator pattern): scores the agent's
    text response against the scenario's ground truth using keyword matching.

    For full episode evaluation: accepts kwargs from the environment's
    internal state (system_healthy, harm_count, progress_state_visits, etc.)
    and delegates to compute_reward().

    Args:
        response: The agent's text response / root cause explanation.
        scenario: Scenario dict with ground truth (root_service, root_cause_type, etc.)
        **kwargs: Episode state overrides (system_healthy, harm_count, etc.)

    Returns:
        Score in [0.01, 0.99].
    """
    if kwargs.get("system_healthy") is not None:
        # Full episode mode — use compute_reward directly
        return _grade_episode(**kwargs)

    # Single-response mode — keyword-based grading
    if not scenario:
        return 0.001

    return _grade_response(response, scenario, difficulty="easy")


def grade_medium(response: str = "", scenario: Optional[Dict] = None, **kwargs) -> float:
    """Grade a medium-tier episode. See grade_easy for details."""
    if kwargs.get("system_healthy") is not None:
        return _grade_episode(**kwargs)
    if not scenario:
        return 0.001
    return _grade_response(response, scenario, difficulty="medium")


def grade_hard(response: str = "", scenario: Optional[Dict] = None, **kwargs) -> float:
    """Grade a hard-tier episode. See grade_easy for details."""
    if kwargs.get("system_healthy") is not None:
        return _grade_episode(**kwargs)
    if not scenario:
        return 0.001
    return _grade_response(response, scenario, difficulty="hard")


def _grade_response(response: str, scenario: Dict, difficulty: str) -> float:
    """Grade a single text response against scenario ground truth.

    Used when the validator calls the grader with just a response string
    (single-step evaluation pattern, like DarDrax's env).

    Scoring:
    - Correct root service mentioned: +0.30
    - Correct failure type mentioned: +0.20
    - Explanation keywords found: up to +0.30
    - Causal explanation present: +0.10
    - Structured/actionable response: +0.10

    Returns score in [0.01, 0.99].
    """
    r = response.lower().strip()
    if not r:
        return 0.001

    score = 0.0
    failure = scenario.get("failure", {})
    root_service = failure.get("root_service", "").lower()
    root_cause_type = failure.get("root_cause_type", "").lower()
    keywords = failure.get("explanation_keywords", [])

    # Root service identification (0.30)
    if root_service and root_service in r:
        score += 0.30

    # Failure type identification (0.20)
    type_aliases = {
        "config_drift": ["config drift", "configuration change", "config change", "misconfiguration"],
        "clock_drift": ["clock drift", "tsc drift", "time drift", "clock skew"],
        "cert_expiry": ["certificate expired", "cert expired", "tls expired", "mtls"],
        "disk_full": ["disk full", "disk exhaustion", "no space", "volume full"],
    }
    type_matched = root_cause_type in r
    for alias_key, aliases in type_aliases.items():
        if alias_key in root_cause_type:
            if any(a in r for a in aliases):
                type_matched = True
    if type_matched:
        score += 0.20

    # Explanation keywords (up to 0.30)
    if keywords:
        hits = sum(1 for kw in keywords if kw.lower() in r)
        keyword_ratio = hits / len(keywords)
        score += 0.30 * keyword_ratio

    # Causal explanation bonus (0.10)
    causal_terms = ["because", "due to", "caused by", "root cause", "resulting from", "leads to"]
    if any(term in r for term in causal_terms):
        score += 0.10

    # Structured response bonus (0.10)
    structured_markers = ["1.", "2.", "step", "first", "then", "finally", "remediation", "fix"]
    if sum(1 for m in structured_markers if m in r) >= 2:
        score += 0.10

    return round(min(0.999, max(0.001, score)), 4)
