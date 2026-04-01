"""Test reward function edge cases and range compliance."""

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


def test_reward_failure_type_match():
    r_match = compute_reward(
        "cause", "svc", "cause", "svc", 5, 10, "medium",
        submitted_failure_type="oom_kill", true_failure_type="oom_kill",
    )
    r_nomatch = compute_reward(
        "cause", "svc", "cause", "svc", 5, 10, "medium",
        submitted_failure_type="config_drift", true_failure_type="oom_kill",
    )
    assert r_match > r_nomatch


def test_reward_causal_chain():
    r_chain = compute_reward(
        "cause", "svc", "cause", "svc", 5, 10, "medium",
        submitted_chain=["svc", "api-gw", "web"],
        all_services={"svc", "api-gw", "web", "db"},
        true_failure_type="oom_kill", submitted_failure_type="oom_kill",
    )
    r_no_chain = compute_reward(
        "cause", "svc", "cause", "svc", 5, 10, "medium",
        true_failure_type="oom_kill", submitted_failure_type="oom_kill",
    )
    assert r_chain > r_no_chain


def test_reward_efficiency_nonlinear():
    r_under = compute_reward("cause", "svc", "cause", "svc", 5, 10, "medium")
    r_full = compute_reward("cause", "svc", "cause", "svc", 10, 10, "medium")
    assert r_under >= r_full


def test_reward_coverage_bonus():
    r_broad = compute_reward(
        "cause", "svc", "true cause", "svc", 5, 10, "medium",
        services_queried={"a", "b", "c", "d"}, all_services={"a", "b", "c", "d", "e"},
    )
    r_narrow = compute_reward(
        "cause", "svc", "true cause", "svc", 5, 10, "medium",
        services_queried={"a"}, all_services={"a", "b", "c", "d", "e"},
    )
    assert r_broad > r_narrow


def test_reward_no_difficulty_multiplier():
    """Difficulty does not affect reward — harder scenarios are inherently harder."""
    kwargs = dict(
        submitted_root_cause="pool exhausted", submitted_service="db",
        true_root_cause="pool exhausted", true_root_service="db",
        steps_used=3, query_budget=10,
    )
    r_easy = compute_reward(**kwargs, difficulty="easy")
    r_hard = compute_reward(**kwargs, difficulty="hard")
    assert r_easy == r_hard


def test_reward_never_exceeds_one():
    r = compute_reward(
        "exact same statement", "exact-service",
        "exact same statement", "exact-service",
        0, 15, "expert", confidence=1.0,
        services_queried={"a", "b", "c"}, all_services={"a", "b", "c"},
        submitted_failure_type="oom_kill", true_failure_type="oom_kill",
        submitted_chain=["exact-service", "a", "b"],
    )
    assert r <= 1.0


def test_reward_zero_budget():
    r = compute_reward("cause", "svc", "cause", "svc", 0, 0, "medium")
    assert 0.0 <= r <= 1.0


def test_reward_confidence_calibration():
    r_high_conf = compute_reward(
        "pool exhausted", "db", "pool exhausted", "db",
        3, 10, "medium", confidence=0.9,
        submitted_failure_type="connection_pool", true_failure_type="connection_pool",
    )
    r_low_conf = compute_reward(
        "pool exhausted", "db", "pool exhausted", "db",
        3, 10, "medium", confidence=0.1,
        submitted_failure_type="connection_pool", true_failure_type="connection_pool",
    )
    assert r_high_conf > r_low_conf
