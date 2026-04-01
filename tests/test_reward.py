"""Test reward function edge cases and range compliance.

V3 reward function: 4 components, all gated on correct service identification.
No coverage, efficiency, or calibration components.
"""

import pytest
from server.reward import compute_reward


def test_reward_range_perfect():
    r = compute_reward(
        "connection pool exhausted due to leak", "db-primary",
        "connection pool exhausted due to leak", "db-primary",
        3, 10, "medium", confidence=0.9,
        services_queried={"db-primary", "api-gw", "payment"},
        all_services={"db-primary", "api-gw", "payment", "auth"},
        submitted_failure_type="connection_pool",
        true_failure_type="connection_pool",
        submitted_chain=["db-primary", "payment", "api-gw"],
    )
    assert 0.0 <= r <= 1.0


def test_reward_range_wrong():
    r = compute_reward(
        "high CPU usage", "web-server",
        "connection pool exhausted", "db-primary",
        10, 10, "medium",
    )
    assert 0.0 <= r <= 1.0


def test_reward_range_empty():
    r = compute_reward("", "", "pool exhausted", "db", 0, 10, "medium")
    assert 0.0 <= r <= 1.0


def test_reward_service_match_contributes():
    r_match = compute_reward("something", "db", "something else", "db", 5, 10, "medium")
    r_nomatch = compute_reward("something", "web", "something else", "db", 5, 10, "medium")
    assert r_match > r_nomatch


def test_reward_wrong_service_scores_zero():
    """Wrong service identification must score exactly 0.0."""
    r = compute_reward(
        "connection pool exhausted due to leak", "wrong-service",
        "connection pool exhausted due to leak", "db-primary",
        3, 10, "medium",
        submitted_failure_type="connection_pool",
        true_failure_type="connection_pool",
        submitted_chain=["wrong-service", "api-gw"],
        all_services={"db-primary", "api-gw", "wrong-service"},
    )
    assert r == 0.0


def test_reward_failure_type_match():
    r_match = compute_reward(
        "cause text here is long enough", "svc",
        "cause text here is long enough", "svc", 5, 10, "medium",
        submitted_failure_type="oom_kill", true_failure_type="oom_kill",
    )
    r_nomatch = compute_reward(
        "cause text here is long enough", "svc",
        "cause text here is long enough", "svc", 5, 10, "medium",
        submitted_failure_type="config_drift", true_failure_type="oom_kill",
    )
    assert r_match > r_nomatch


def test_reward_failure_type_gated_on_service():
    """Failure type should not contribute if service is wrong."""
    r = compute_reward(
        "cause text", "wrong-svc",
        "cause text", "real-svc", 5, 10, "medium",
        submitted_failure_type="oom_kill", true_failure_type="oom_kill",
    )
    assert r == 0.0


def test_reward_causal_chain():
    r_chain = compute_reward(
        "cause text here is long enough", "svc",
        "cause text here is long enough", "svc", 5, 10, "medium",
        submitted_chain=["svc", "api-gw", "web"],
        all_services={"svc", "api-gw", "web", "db"},
        true_failure_type="oom_kill", submitted_failure_type="oom_kill",
    )
    r_no_chain = compute_reward(
        "cause text here is long enough", "svc",
        "cause text here is long enough", "svc", 5, 10, "medium",
        true_failure_type="oom_kill", submitted_failure_type="oom_kill",
    )
    assert r_chain > r_no_chain


def test_reward_causal_chain_gated_on_service():
    """Causal chain should not contribute if service is wrong."""
    r = compute_reward(
        "cause text", "wrong-svc",
        "cause text", "real-svc", 5, 10, "medium",
        submitted_chain=["wrong-svc", "api-gw"],
        all_services={"wrong-svc", "api-gw", "real-svc"},
    )
    assert r == 0.0


def test_reward_causal_chain_graph_validation():
    """Chain edges must match real dependency graph when provided."""
    graph = {
        "root-db": {"upstream": []},
        "mid-svc": {"upstream": ["root-db"]},
        "api-gw": {"upstream": ["mid-svc"]},
    }
    # Valid chain: root-db -> mid-svc -> api-gw
    r_valid = compute_reward(
        "cause text here is long enough", "root-db",
        "cause text here is long enough", "root-db", 5, 10, "medium",
        submitted_chain=["root-db", "mid-svc", "api-gw"],
        all_services={"root-db", "mid-svc", "api-gw"},
        services_graph=graph,
    )
    # Invalid chain: root-db -> api-gw (skips mid-svc, no direct edge)
    r_invalid = compute_reward(
        "cause text here is long enough", "root-db",
        "cause text here is long enough", "root-db", 5, 10, "medium",
        submitted_chain=["root-db", "api-gw"],
        all_services={"root-db", "mid-svc", "api-gw"},
        services_graph=graph,
    )
    assert r_valid > r_invalid


def test_reward_no_difficulty_multiplier():
    """Difficulty does not affect reward -- harder scenarios are inherently harder."""
    kwargs = dict(
        submitted_root_cause="pool exhausted due to connection leak",
        submitted_service="db",
        true_root_cause="pool exhausted due to connection leak",
        true_root_service="db",
        steps_used=3, query_budget=10,
    )
    r_easy = compute_reward(**kwargs, difficulty="easy")
    r_hard = compute_reward(**kwargs, difficulty="hard")
    assert r_easy == r_hard


def test_reward_never_exceeds_one():
    r = compute_reward(
        "exact same statement with enough length", "exact-service",
        "exact same statement with enough length", "exact-service",
        0, 15, "expert", confidence=1.0,
        services_queried={"a", "b", "c"}, all_services={"a", "b", "c"},
        submitted_failure_type="oom_kill", true_failure_type="oom_kill",
        submitted_chain=["exact-service", "a", "b"],
    )
    assert r <= 1.0


def test_reward_zero_budget():
    r = compute_reward(
        "cause text", "svc", "cause text", "svc", 0, 0, "medium"
    )
    assert 0.0 <= r <= 1.0


def test_reward_semantic_gated_on_service():
    """Semantic similarity should not contribute if service is wrong."""
    r = compute_reward(
        "connection pool exhausted due to leak", "wrong-svc",
        "connection pool exhausted due to leak", "correct-svc",
        3, 10, "medium",
    )
    assert r == 0.0


def test_reward_length_penalty_short():
    """Very short explanations should score less than proper ones."""
    r_short = compute_reward(
        "oom", "svc", "worker pool OOM killed due to memory leak", "svc",
        5, 10, "medium",
    )
    r_proper = compute_reward(
        "worker pool OOM killed due to memory leak causing restarts", "svc",
        "worker pool OOM killed due to memory leak", "svc",
        5, 10, "medium",
    )
    # Short (3 chars) gets length_factor=0, so semantic_score=0
    # Proper length gets nonzero semantic_score
    assert r_proper > r_short


def test_reward_gradient_exists():
    """Reward must provide gradient: correct service < +type < +explanation."""
    # Service only
    r1 = compute_reward(
        "totally wrong explanation", "db",
        "connection pool exhausted due to leak in db", "db",
        5, 10, "medium",
        submitted_failure_type="wrong_type", true_failure_type="connection_pool",
    )
    # Service + type
    r2 = compute_reward(
        "totally wrong explanation", "db",
        "connection pool exhausted due to leak in db", "db",
        5, 10, "medium",
        submitted_failure_type="connection_pool", true_failure_type="connection_pool",
    )
    # Service + type + good explanation
    r3 = compute_reward(
        "connection pool in db service exhausted due to a connection leak", "db",
        "connection pool exhausted due to leak in db", "db",
        5, 10, "medium",
        submitted_failure_type="connection_pool", true_failure_type="connection_pool",
    )
    assert r1 < r2 <= r3
    assert r1 == 0.30  # service match only
    assert r2 == 0.50  # service + type match
