"""Test V4 reward function — ungameable + learnable.

6 components:
  Tier 1 (Diagnosis, 0.70): service ID, failure type, explanation, causal chain
  Tier 2 (Investigation, 0.30): efficiency, breadth
"""

import pytest
from server.reward import compute_reward


GRAPH = {
    "root-db": {"upstream": []},
    "mid-svc": {"upstream": ["root-db"]},
    "api-gw": {"upstream": ["mid-svc"]},
    "unrelated": {"upstream": ["api-gw"]},
}

BASE = dict(
    submitted_root_cause="root-db connection pool exhausted due to config drift",
    submitted_service="root-db",
    true_root_cause="root-db connection pool exhausted due to config drift in v2.3",
    true_root_service="root-db",
    steps_used=5, query_budget=10, difficulty="medium",
    all_services={"root-db", "mid-svc", "api-gw", "unrelated"},
    services_graph=GRAPH,
)


# ─── Component 1: Service ID ───────────────────────────────────────

def test_service_exact_match():
    r = compute_reward(**{**BASE})
    assert r >= 0.25  # at least service score


def test_service_adjacent_partial():
    """One-hop neighbor gets 0.08, not 0.0."""
    r = compute_reward(**{**BASE, "submitted_service": "mid-svc"})
    assert 0.05 < r < 0.25  # adjacency bonus + investigation scores


def test_service_wrong_scores_near_zero():
    """Wrong non-adjacent service: only investigation tier contributes."""
    r = compute_reward(**{**BASE, "submitted_service": "unrelated"})
    assert r < 0.20  # no diagnosis credit, only investigation


def test_service_wrong_gates_everything():
    """Wrong service → type/explanation/chain all zero."""
    r_wrong = compute_reward(**{
        **BASE, "submitted_service": "wrong",
        "submitted_failure_type": "config_drift",
        "true_failure_type": "config_drift",
        "submitted_chain": ["wrong", "mid-svc"],
    })
    r_right = compute_reward(**{
        **BASE,
        "submitted_failure_type": "config_drift",
        "true_failure_type": "config_drift",
        "submitted_chain": ["root-db", "mid-svc"],
    })
    assert r_right > r_wrong + 0.30  # at least 0.30 more


# ─── Component 2: Failure Type ──────────────────────────────────────

def test_failure_type_match():
    r_match = compute_reward(**{**BASE,
        "submitted_failure_type": "config_drift", "true_failure_type": "config_drift"})
    r_no = compute_reward(**{**BASE,
        "submitted_failure_type": "oom_kill", "true_failure_type": "config_drift"})
    assert r_match > r_no


def test_failure_type_gated_on_service():
    r = compute_reward(**{**BASE, "submitted_service": "wrong",
        "submitted_failure_type": "config_drift", "true_failure_type": "config_drift"})
    # Should not get type credit
    r2 = compute_reward(**{**BASE, "submitted_service": "wrong"})
    assert r == r2  # no difference — type is gated


# ─── Component 3: Explanation ────────────────────────────────────────

def test_explanation_gradient():
    r_good = compute_reward(**{**BASE,
        "submitted_root_cause": "root-db connection pool exhausted due to config drift in deployment v2.3"})
    r_bad = compute_reward(**{**BASE,
        "submitted_root_cause": "something completely different and wrong explanation"})
    assert r_good > r_bad


def test_explanation_length_penalty():
    r_short = compute_reward(**{**BASE, "submitted_root_cause": "oom"})
    r_proper = compute_reward(**{**BASE})
    assert r_proper > r_short


def test_explanation_gated_on_service():
    r = compute_reward(**{**BASE, "submitted_service": "wrong"})
    # Even perfect explanation, wrong service → no explanation credit
    assert r < 0.15


# ─── Component 4: Causal Chain ───────────────────────────────────────

def test_chain_valid_edges():
    r = compute_reward(**{**BASE,
        "submitted_chain": ["root-db", "mid-svc", "api-gw"]})
    r_no = compute_reward(**{**BASE})
    assert r > r_no


def test_chain_partial_edges():
    """Edge-fraction: 1/2 valid edges should score less than 2/2."""
    r_full = compute_reward(**{**BASE,
        "submitted_chain": ["root-db", "mid-svc", "api-gw"]})  # 2/2 valid
    r_partial = compute_reward(**{**BASE,
        "submitted_chain": ["root-db", "api-gw"]})  # 0/1 valid (skips mid)
    assert r_full > r_partial


def test_chain_random_services_score_zero():
    """Random chain not starting with root → 0."""
    r = compute_reward(**{**BASE,
        "submitted_chain": ["mid-svc", "api-gw", "unrelated"]})
    r_no = compute_reward(**{**BASE})
    assert r == r_no  # no chain credit


# ─── Component 5: Investigation Efficiency ───────────────────────────

def test_efficiency_no_waste():
    r = compute_reward(**{**BASE,
        "tool_call_history": [
            ("read_logs", "root-db"), ("check_metric", "root-db"),
            ("read_logs", "mid-svc"), ("check_metric", "mid-svc"),
        ]})
    r_no_hist = compute_reward(**{**BASE, "tool_call_history": []})
    assert r > r_no_hist


def test_efficiency_penalizes_duplicates():
    r_clean = compute_reward(**{**BASE,
        "tool_call_history": [
            ("read_logs", "root-db"), ("check_metric", "root-db"),
            ("read_logs", "mid-svc"),
        ]})
    r_dupes = compute_reward(**{**BASE,
        "tool_call_history": [
            ("read_logs", "root-db"), ("read_logs", "root-db"),
            ("read_logs", "root-db"),
        ]})
    assert r_clean > r_dupes


def test_efficiency_requires_min_investigation():
    """Submitting after 0-1 queries = rushed = 0 efficiency."""
    r = compute_reward(**{**BASE,
        "tool_call_history": [("read_logs", "root-db")]})
    r2 = compute_reward(**{**BASE, "tool_call_history": []})
    assert r == r2  # both get 0 efficiency (< 2 calls)


def test_efficiency_not_gated_on_service():
    """Efficiency rewards process even with wrong diagnosis."""
    r = compute_reward(**{**BASE, "submitted_service": "wrong",
        "tool_call_history": [
            ("read_logs", "root-db"), ("check_metric", "root-db"),
            ("read_logs", "mid-svc"),
        ]})
    assert r > 0.0  # gets efficiency + breadth even with wrong service


# ─── Component 6: Investigation Breadth ──────────────────────────────

def test_breadth_rewards_relevant():
    r_relevant = compute_reward(**{**BASE,
        "services_queried": {"root-db", "mid-svc", "api-gw"},
        "tool_call_history": [("read_logs", "root-db"), ("read_logs", "mid-svc"), ("read_logs", "api-gw")]})
    r_irrelevant = compute_reward(**{**BASE,
        "services_queried": {"unrelated"},
        "tool_call_history": [("read_logs", "unrelated"), ("read_logs", "unrelated")]})
    assert r_relevant > r_irrelevant


def test_breadth_focused_beats_unfocused():
    """Querying relevant services scores better than querying irrelevant ones."""
    history = [("read_logs", "root-db"), ("check_metric", "root-db"),
               ("read_logs", "mid-svc"), ("check_metric", "mid-svc")]
    # Focused: queried causal chain services
    r_focused = compute_reward(**{**BASE,
        "services_queried": {"root-db", "mid-svc"},
        "tool_call_history": history})
    # Unfocused: queried only irrelevant services
    history_bad = [("read_logs", "unrelated"), ("check_metric", "unrelated"),
                   ("read_logs", "unrelated"), ("check_metric", "unrelated")]
    r_unfocused = compute_reward(**{**BASE,
        "services_queried": {"unrelated"},
        "tool_call_history": history_bad})
    assert r_focused > r_unfocused


def test_breadth_not_gated_on_service():
    r = compute_reward(**{**BASE, "submitted_service": "wrong",
        "services_queried": {"root-db", "mid-svc"},
        "tool_call_history": [("read_logs", "root-db"), ("read_logs", "mid-svc")]})
    assert r > 0.0


# ─── Overall Properties ─────────────────────────────────────────────

def test_reward_range():
    for svc in ["root-db", "wrong", "mid-svc"]:
        r = compute_reward(**{**BASE, "submitted_service": svc})
        assert 0.0 <= r <= 1.0


def test_reward_never_exceeds_one():
    r = compute_reward(**{**BASE,
        "submitted_failure_type": "config_drift", "true_failure_type": "config_drift",
        "submitted_chain": ["root-db", "mid-svc", "api-gw"],
        "services_queried": {"root-db", "mid-svc", "api-gw"},
        "tool_call_history": [("read_logs", "root-db"), ("check_metric", "root-db"),
                              ("read_logs", "mid-svc"), ("check_metric", "api-gw")]})
    assert r <= 1.0


def test_gradient_exists():
    """Reward must increase: wrong < adjacent < correct < +type < +explanation."""
    r_wrong = compute_reward(**{**BASE, "submitted_service": "wrong", "tool_call_history": []})
    r_adj = compute_reward(**{**BASE, "submitted_service": "mid-svc", "tool_call_history": []})
    r_svc = compute_reward(**{**BASE, "tool_call_history": []})
    r_type = compute_reward(**{**BASE, "submitted_failure_type": "config_drift",
        "true_failure_type": "config_drift", "tool_call_history": []})
    assert r_wrong < r_adj < r_svc < r_type
