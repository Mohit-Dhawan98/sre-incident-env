"""Reward computation for SRE incident remediation (V3 maze navigation).

7 components, sum to 1.00 max when fully solved, 0.50 max when not:

  1. EXIT             (0.40) — binary, healthy only
  2. PROGRESS         (0.20) — quadratic, continuous
  3. DIAGNOSIS        (0.10) — gated on healthy
  4. EFFICIENCY       (0.10) — gated on progress
  5. DISCIPLINE       (0.10) — gated on progress
  6. TRAP AVOIDANCE   (0.05) — always
  7. DIVERSITY        (0.05) — gated on progress

All deterministic. No model loading. Perfect run = 1.00.
"""

from typing import Any, Dict, List, Set


def _get_neighbors(service: str, services_graph: Dict[str, Any]) -> Set[str]:
    """Get services one hop away in the dependency graph."""
    if not services_graph:
        return set()
    svc_lower = service.strip().lower()
    graph_lower = {}
    for name, info in services_graph.items():
        graph_lower[name.lower()] = [u.lower() for u in info.get("upstream", [])]

    neighbors = set()
    for name, upstreams in graph_lower.items():
        if name == svc_lower:
            neighbors.update(upstreams)
        if svc_lower in upstreams:
            neighbors.add(name)
    return neighbors


def compute_reward(
    submitted_root_cause: str,
    submitted_service: str,
    true_root_service: str,
    submitted_failure_type: str,
    true_failure_type: str,
    services_graph: Dict[str, Any],
    explanation_keywords: List[str],
    system_healthy: bool,
    harm_count: int,
    optimal_steps: int,
    remediation_count: int,
    observation_after_fix: int,
    discovered_before_action: int,
    execute_runbook_count: int,
    unique_remediation_count: int,
    progress_state_visits: int,
) -> float:
    """V3 maze-navigation reward. Returns float in (0.0001, 0.9999)."""

    # ── 1. EXIT (0.40) — binary, healthy only ────────────────────
    maze_exit = 0.40 if system_healthy else 0.0

    # ── 2. PROGRESS (0.20) — quadratic continuous credit ─────────
    if optimal_steps > 0:
        progress_ratio = min(1.0, progress_state_visits / optimal_steps)
        maze_progress = 0.20 * (progress_ratio ** 2)
    else:
        maze_progress = 0.0

    # ── 3. DIAGNOSIS (0.10) — gated on system_healthy ────────────
    if system_healthy:
        sub_svc = submitted_service.strip().lower()
        true_svc = true_root_service.strip().lower()

        if sub_svc == true_svc:
            diag_service = 0.05
        elif services_graph and sub_svc in _get_neighbors(true_svc, services_graph):
            diag_service = 0.02
        else:
            diag_service = 0.0

        diag_type = 0.025 if (
            submitted_failure_type.strip().lower() == true_failure_type.strip().lower()
            and submitted_failure_type.strip()
        ) else 0.0

        submitted_text = submitted_root_cause.strip().lower()
        if explanation_keywords and len(submitted_text) >= 10:
            found = sum(1 for kw in explanation_keywords if kw.lower() in submitted_text)
            diag_keywords = (found / len(explanation_keywords)) * 0.025
        else:
            diag_keywords = 0.0

        maze_diagnosis = diag_service + diag_type + diag_keywords
    else:
        maze_diagnosis = 0.0

    # ── 4. EFFICIENCY (0.10) — gated on progress ─────────────────
    if remediation_count == 0 or optimal_steps == 0:
        maze_efficiency = 0.0
    else:
        progress_scale = min(1.0, progress_state_visits / optimal_steps)
        ratio = min(1.0, optimal_steps / remediation_count)
        maze_efficiency = 0.10 * ratio * progress_scale

    # ── 5. DISCIPLINE (0.10) — gated on progress ─────────────────
    maze_discipline = 0.0
    if remediation_count > 0 and progress_state_visits > 0 and optimal_steps > 0:
        progress_scale = min(1.0, progress_state_visits / optimal_steps)
        observe_ratio = min(1.0, observation_after_fix / remediation_count)
        maze_discipline += 0.05 * observe_ratio * progress_scale
        if execute_runbook_count > 0:
            discover_ratio = min(1.0, discovered_before_action / execute_runbook_count)
            maze_discipline += 0.05 * discover_ratio * progress_scale
        else:
            maze_discipline += 0.05 * progress_scale

    # ── 6. TRAP AVOIDANCE (0.05) — always ────────────────────────
    maze_traps = max(0.0, 0.05 - harm_count * 0.025)

    # ── 7. DIVERSITY (0.05) — gated on progress ──────────────────
    if remediation_count > 0 and progress_state_visits > 0 and optimal_steps > 0:
        progress_scale = min(1.0, progress_state_visits / optimal_steps)
        unique_ratio = min(1.0, unique_remediation_count / remediation_count)
        maze_unique = 0.05 * unique_ratio * progress_scale
    else:
        maze_unique = 0.0

    # Healthy max:    0.40 + 0.20 + 0.10 + 0.10 + 0.10 + 0.05 + 0.05 = 1.00
    # Not-healthy max: 0   + 0.20 + 0    + 0.10 + 0.10 + 0.05 + 0.05 = 0.50
    total = (
        maze_exit + maze_progress + maze_diagnosis
        + maze_efficiency + maze_discipline + maze_traps + maze_unique
    )

    # Hackathon Phase 2: scores must be strictly in (0, 1)
    return round(min(0.999, max(0.001, total)), 4)
