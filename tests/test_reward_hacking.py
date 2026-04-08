"""Adversarial reward-hacking tests.

Verifies the V3 reward function cannot be gamed by common exploits:

  1. Diagnosis-only with zero remediation
  2. Spam the same correct action repeatedly
  3. Correct diagnosis but wrong remediation path
  4. Verify_resolution with empty / hallucinated root cause
  5. Trap-farming (racking up many worsened outcomes)
  6. Breadth gaming (discovering every service but fixing nothing)
  7. Partial-path gaming (stopping at the first progress step)
  8. Discipline inflation (observe-after-fix without real progress)

If any of these score above the expected cap, the reward function has a hole.
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
    optimal_steps=4,
)


def _call(**overrides):
    defaults = dict(
        system_healthy=False,
        harm_count=0,
        remediation_count=0,
        observation_after_fix=0,
        discovered_before_action=0,
        execute_runbook_count=0,
        unique_remediation_count=0,
        progress_state_visits=0,
    )
    return compute_reward(**{**BASE, **defaults, **overrides})


# ── Exploit 1: Diagnosis-only without any remediation ──────────────

def test_diagnosis_only_is_capped():
    """Perfect diagnosis but zero remediation must score near floor.

    Historical bug: V1 reward credited investigation + diagnosis regardless
    of whether the system was fixed. An agent could call submit_diagnosis
    and get 0.70 without running a single remediation. V3 gates diagnosis
    credit on system_healthy.
    """
    r = _call()  # perfect diagnosis args, but no remediation, not healthy
    # Only trap_avoidance (0.05) survives. Diagnosis gated off. Progress=0.
    assert r < 0.10, f"diagnosis-only exploit scored {r}, should be < 0.10"


# ── Exploit 2: Spam a single correct action repeatedly ──────────────

def test_spam_correct_action_does_not_inflate():
    """Calling the same correct first-step action N times stays at depth 1.

    The state machine caps max_progress_depth at the furthest point reached,
    so re-hitting the same progress edge does not grow progress_state_visits.
    """
    # Agent "solved" step 1 once, then spammed it 20 more times.
    r_once = _call(
        progress_state_visits=1, remediation_count=1,
        observation_after_fix=1, execute_runbook_count=1,
        discovered_before_action=1, unique_remediation_count=1,
    )
    r_spam = _call(
        progress_state_visits=1,      # still depth 1 — spam doesn't advance
        remediation_count=21,         # 21 calls
        observation_after_fix=1,      # didn't observe every time
        execute_runbook_count=21,
        discovered_before_action=1,
        unique_remediation_count=1,   # all identical
    )
    # Spamming should NOT score higher — efficiency/diversity/discipline penalize it.
    assert r_spam <= r_once + 0.01, f"spam {r_spam} beats clean {r_once}"


def test_spam_correct_action_kills_efficiency():
    """Efficiency ratio collapses when remediation_count >> optimal_steps."""
    r_clean = _call(
        progress_state_visits=4, remediation_count=4,
        observation_after_fix=4, execute_runbook_count=4,
        discovered_before_action=4, unique_remediation_count=4,
    )
    r_wasteful = _call(
        progress_state_visits=4, remediation_count=40,
        observation_after_fix=4, execute_runbook_count=40,
        discovered_before_action=4, unique_remediation_count=4,
    )
    # Wasteful run should lose efficiency + diversity + discipline.
    assert r_clean - r_wasteful > 0.10, \
        f"waste gap too small: clean={r_clean} wasteful={r_wasteful}"


# ── Exploit 3: Wrong service but correct failure_type/keywords ──────

def test_wrong_service_blocks_diagnosis_credit():
    """Diagnosis service component is strict: non-neighbor service = 0."""
    r = _call(
        system_healthy=True,
        submitted_service="api-gw",  # 2 hops away, not adjacent
        progress_state_visits=4, remediation_count=4,
        observation_after_fix=4, execute_runbook_count=4,
        discovered_before_action=4, unique_remediation_count=4,
    )
    # api-gw is not a direct neighbor of root-db in failure propagation.
    # Loses the 0.05 service credit. Type + keywords still score.
    # Max: 0.40 + 0.20 + (0 + 0.025 + 0.025) + 0.10 + 0.10 + 0.05 + 0.05 = 0.95
    assert r < 0.96


# ── Exploit 4: Empty / hallucinated root cause text ────────────────

def test_empty_root_cause_blocks_keyword_credit():
    """Keyword matching requires non-trivial root_cause text."""
    r = _call(
        system_healthy=True,
        submitted_root_cause="x",  # too short, keyword match skipped
        progress_state_visits=4, remediation_count=4,
        observation_after_fix=4, execute_runbook_count=4,
        discovered_before_action=4, unique_remediation_count=4,
    )
    # Loses the 0.025 keyword credit. diag = 0.05 + 0.025 + 0 = 0.075
    # Total = 0.40 + 0.20 + 0.075 + 0.10 + 0.10 + 0.05 + 0.05 = 0.975
    assert 0.97 < r < 0.98


# ── Exploit 5: Trap farming (worsened actions should only hurt) ─────

def test_trap_actions_reduce_reward():
    """Each worsened outcome costs 0.025 from trap avoidance, floored at 0."""
    base = _call(progress_state_visits=2, remediation_count=2,
                 observation_after_fix=2, execute_runbook_count=2,
                 discovered_before_action=2, unique_remediation_count=2)
    with_traps = _call(harm_count=3,
                       progress_state_visits=2, remediation_count=2,
                       observation_after_fix=2, execute_runbook_count=2,
                       discovered_before_action=2, unique_remediation_count=2)
    # 3 traps fully drains the 0.05 trap avoidance bucket
    assert base - with_traps >= 0.04, \
        f"trap penalty too weak: base={base} traps={with_traps}"


def test_trap_avoidance_cannot_go_negative():
    """Maxing out harm_count must not push the component below zero."""
    r = _call(harm_count=100)
    assert r >= 0.0001  # clamped to valid phase-2 range


# ── Exploit 6: Partial-path gaming (hit step 1, then verify) ────────

def test_partial_progress_is_quadratically_penalized():
    """Quadratic curve makes early partial progress worth much less.

    An agent that reaches step 1 of a 4-step scenario and then verifies
    should get substantially less than an agent that reaches step 2.
    """
    r1 = _call(progress_state_visits=1, remediation_count=1,
               observation_after_fix=1, execute_runbook_count=1,
               discovered_before_action=1, unique_remediation_count=1)
    r2 = _call(progress_state_visits=2, remediation_count=2,
               observation_after_fix=2, execute_runbook_count=2,
               discovered_before_action=2, unique_remediation_count=2)
    r3 = _call(progress_state_visits=3, remediation_count=3,
               observation_after_fix=3, execute_runbook_count=3,
               discovered_before_action=3, unique_remediation_count=3)
    # Progress component: (1/4)²=0.0125, (2/4)²=0.05, (3/4)²=0.1125 of 0.20
    # Efficiency + discipline + diversity scale linearly with progress_scale.
    # Gap between r1 and r2 should be meaningful.
    assert r2 - r1 > 0.03
    assert r3 - r2 > 0.04


# ── Exploit 7: Discipline inflation (observe/discover without fix) ──

def test_discipline_requires_progress():
    """Discipline is gated on progress_state_visits > 0.

    An agent that observes after every no-op remediation should not farm
    discipline credit without actually fixing anything.
    """
    r = _call(
        remediation_count=20,
        observation_after_fix=20,     # "I observed after every action!"
        execute_runbook_count=20,
        discovered_before_action=20,  # "I discovered every action!"
        unique_remediation_count=20,  # "I never repeated!"
        # BUT: progress_state_visits=0 (nothing actually worked)
    )
    # Only trap avoidance survives. Discipline + diversity both gated off.
    assert r < 0.10, f"discipline inflation scored {r}"


# ── Exploit 8: Verify without calling any remediation ──────────────

def test_zero_remediation_floors():
    """remediation_count=0 zeroes efficiency, discipline, diversity."""
    r = _call(
        system_healthy=True,  # even if they lied about being healthy
        remediation_count=0,
        progress_state_visits=0,
    )
    # Exit 0.40 + Progress 0 + Diagnosis 0.10 + Eff 0 + Disc 0 + Traps 0.05 + Div 0
    # = 0.55 — still bounded well below full credit
    assert r < 0.60


# ── Sanity: perfect run still attainable ──────────────────────────

def test_perfect_run_reaches_cap():
    """None of the above caps prevent a legitimate perfect run from scoring ~1.0."""
    r = _call(
        system_healthy=True,
        progress_state_visits=4, remediation_count=4,
        observation_after_fix=4, execute_runbook_count=4,
        discovered_before_action=4, unique_remediation_count=4,
    )
    assert r == 0.9999  # clamped max


# ── Bounds check: reward strictly in (0, 1) ───────────────────────

def test_reward_strictly_bounded():
    """Phase 2 validator requires every reward strictly in (0, 1)."""
    import itertools
    for healthy, harm, prog, remed in itertools.product(
        [False, True], [0, 5], [0, 2, 4], [0, 4, 40]
    ):
        r = _call(system_healthy=healthy, harm_count=harm,
                  progress_state_visits=prog, remediation_count=remed,
                  observation_after_fix=remed, execute_runbook_count=remed,
                  discovered_before_action=remed, unique_remediation_count=remed)
        assert 0.0 < r < 1.0, f"out of range: {r} @ h={healthy} harm={harm} p={prog} r={remed}"
