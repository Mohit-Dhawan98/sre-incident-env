"""Test V3 maze-navigation reward function.

7 components, perfect run = 1.00:
  1. Exit (0.40, healthy only)
  2. Progress (0.20, quadratic)
  3. Diagnosis (0.10, gated on healthy)
  4. Efficiency (0.10, gated on progress)
  5. Discipline (0.10, gated on progress)
  6. Trap Avoidance (0.05)
  7. Diversity (0.05, gated on progress)
"""

from server.reward import compute_reward


GRAPH = {
    "root-db": {"upstream": ["mid-svc"]},
    "mid-svc": {"upstream": ["api-gw"]},
    "api-gw": {"upstream": []},
}

BASE = dict(
    submitted_root_cause="root-db connection pool exhausted due to config drift",
    submitted_service="root-db",
    true_root_service="root-db",
    submitted_failure_type="config_drift",
    true_failure_type="config_drift",
    services_graph=GRAPH,
    explanation_keywords=["root-db", "connection pool", "config drift"],
    optimal_steps=3,
)


def _call(**overrides):
    return compute_reward(**{
        **BASE,
        "system_healthy": False,
        "harm_count": 0,
        "remediation_count": 0,
        "observation_after_fix": 0,
        "discovered_before_action": 0,
        "execute_runbook_count": 0,
        "unique_remediation_count": 0,
        "progress_state_visits": 0,
        **overrides,
    })


def test_zero_progress_floors():
    """No progress + not healthy = base trap credit only."""
    r = _call()
    assert 0.04 < r < 0.06


def test_partial_progress_quadratic():
    """1/3 progress = (1/3)² = 0.111 x 0.20 ≈ 0.022 progress credit."""
    r = _call(progress_state_visits=1, remediation_count=1,
              observation_after_fix=1, execute_runbook_count=1,
              discovered_before_action=1, unique_remediation_count=1)
    assert 0.05 < r < 0.25


def test_full_progress_not_healthy():
    """All 3 steps done but not healthy: capped at 0.50."""
    r = _call(progress_state_visits=3, remediation_count=3,
              observation_after_fix=3, execute_runbook_count=3,
              discovered_before_action=3, unique_remediation_count=3)
    assert 0.45 < r < 0.51


def test_perfect_run():
    """Healthy + correct diagnosis + optimal steps = 1.00 (clamped 0.9999)."""
    r = _call(system_healthy=True,
              progress_state_visits=3, remediation_count=3,
              observation_after_fix=3, execute_runbook_count=3,
              discovered_before_action=3, unique_remediation_count=3)
    assert r == 0.9999


def test_trap_penalty():
    """Each worsened outcome costs 0.025 from trap avoidance."""
    r_clean = _call()
    r_one_trap = _call(harm_count=1)
    r_two_traps = _call(harm_count=2)
    assert r_clean - r_one_trap > 0.02
    assert r_one_trap - r_two_traps > 0.02


def test_wrong_service_no_diagnosis_credit():
    """Wrong service = no diagnosis credit even if healthy."""
    r = _call(system_healthy=True, submitted_service="api-gw",
              progress_state_visits=3, remediation_count=3,
              observation_after_fix=3, execute_runbook_count=3,
              discovered_before_action=3, unique_remediation_count=3)
    # api-gw is 2 hops from root-db in failure propagation, so not adjacent.
    # Diagnosis loses service credit (0.05) but keeps type (0.025) + keywords (0.025) = 0.05
    # Exit 0.40 + Progress 0.20 + Diagnosis 0.05 + Eff 0.10 + Disc 0.10 + Traps 0.05 + Div 0.05 = 0.95
    assert 0.93 < r < 0.97


def test_reward_strictly_in_valid_range():
    """Phase 2 validator requires reward strictly in (0, 1)."""
    r = _call()
    assert 0.0 < r < 1.0
    r_perfect = _call(system_healthy=True,
                      progress_state_visits=3, remediation_count=3,
                      observation_after_fix=3, execute_runbook_count=3,
                      discovered_before_action=3, unique_remediation_count=3)
    assert 0.0 < r_perfect < 1.0
