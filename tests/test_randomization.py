"""Test parameter randomization for anti-memorization.

Verifies that seeded randomization:
1. Preserves state-graph topology (invariant across seeds)
2. Changes surface features (service names, config values)
3. Is deterministic (same seed = same output)
4. Keeps domain-locked services fixed
5. Doesn't break the reward function or state machine
"""

import json
import copy
import pytest
from pathlib import Path
from server.scenario_resolver import resolve_scenario, verify_resolution
from server.scenario_loader import ScenarioLoader

SCENARIOS_FILE = str(Path(__file__).resolve().parent.parent / "scenarios" / "incidents_v3.jsonl")

DOMAIN_LOCKED = {
    "zookeeper", "postgres-primary", "etcd-cluster", "kafka-broker",
    "classloader-agent", "tsc-firmware", "haproxy-lb", "nginx-ingress",
    "sysctl-agent", "cert-manager", "numa-balancer", "matching-engine",
    "wal-archiver", "k8s-apiserver", "kube-scheduler", "backup-agent",
    "partition-manager",
}


@pytest.fixture
def loader():
    return ScenarioLoader(SCENARIOS_FILE)


@pytest.fixture
def all_scenarios():
    scenarios = []
    with open(SCENARIOS_FILE) as f:
        for line in f:
            scenarios.append(json.loads(line))
    return scenarios


def test_seed_zero_is_original(all_scenarios):
    """seed=0 returns the original scenario unchanged."""
    for s in all_scenarios:
        resolved = resolve_scenario(s, seed=0)
        # Services should be identical
        assert set(resolved["services"].keys()) == set(
            k for k in s["services"].keys()
        ), f"{s['id']}: seed=0 should not change service names"
        # root_service unchanged
        assert resolved["failure"]["root_service"] == s["failure"]["root_service"]
        # template_vars stripped
        assert "template_vars" not in resolved


def test_seed_deterministic(all_scenarios):
    """Same seed produces identical output."""
    for s in all_scenarios:
        r1 = resolve_scenario(s, seed=42)
        r2 = resolve_scenario(s, seed=42)
        assert r1 == r2, f"{s['id']}: seed=42 produced different outputs on two calls"


def test_seed_varies(all_scenarios):
    """Different seeds produce different outputs (for scenarios with template_vars)."""
    for s in all_scenarios:
        if not s.get("template_vars"):
            continue
        r42 = resolve_scenario(s, seed=42)
        r99 = resolve_scenario(s, seed=99)
        r137 = resolve_scenario(s, seed=137)
        # At least one pair should differ in services
        svcs = [set(r["services"].keys()) for r in [r42, r99, r137]]
        assert len(set(frozenset(s) for s in svcs)) > 1, (
            f"{s['id']}: 3 different seeds all produced identical service names"
        )


def test_topology_invariant(all_scenarios):
    """State names, edge count, and optimal_steps must be identical across seeds."""
    for s in all_scenarios:
        r0 = resolve_scenario(s, seed=0)
        r42 = resolve_scenario(s, seed=42)
        r137 = resolve_scenario(s, seed=137)

        for resolved in [r0, r42, r137]:
            # State names
            orig_states = set(s["failure"]["remediation"]["states"].keys())
            res_states = set(resolved["failure"]["remediation"]["states"].keys())
            assert orig_states == res_states, (
                f"{s['id']}: state names changed: {orig_states} vs {res_states}"
            )
            # optimal_steps
            assert (
                resolved["failure"]["remediation"]["optimal_steps"]
                == s["failure"]["remediation"]["optimal_steps"]
            )
            # Edge count per state
            for state_name in orig_states:
                orig_actions = len(
                    s["failure"]["remediation"]["states"][state_name].get("actions", [])
                )
                res_actions = len(
                    resolved["failure"]["remediation"]["states"][state_name].get(
                        "actions", []
                    )
                )
                assert orig_actions == res_actions, (
                    f"{s['id']}.{state_name}: action count changed {orig_actions} → {res_actions}"
                )


def test_domain_locked_preserved(all_scenarios):
    """Domain-locked service names must NOT be changed by randomization."""
    for s in all_scenarios:
        orig_svcs = set(s["services"].keys())
        locked_in_scenario = orig_svcs & DOMAIN_LOCKED

        for seed in [42, 99, 137]:
            resolved = resolve_scenario(s, seed=seed)
            res_svcs = set(resolved["services"].keys())
            for locked in locked_in_scenario:
                assert locked in res_svcs, (
                    f"{s['id']} seed={seed}: domain-locked '{locked}' was removed"
                )


def test_verify_resolution_passes(all_scenarios):
    """verify_resolution() should find zero issues across all seeds."""
    for s in all_scenarios:
        for seed in [0, 42, 99]:
            resolved = resolve_scenario(s, seed=seed)
            issues = verify_resolution(s, resolved, domain_locked=DOMAIN_LOCKED)
            assert issues == [], (
                f"{s['id']} seed={seed}: verification issues: {issues}"
            )


def test_action_targets_exist_in_services(all_scenarios):
    """Every state machine action target must exist in the resolved services dict."""
    for s in all_scenarios:
        for seed in [42, 137]:
            resolved = resolve_scenario(s, seed=seed)
            res_svcs = set(resolved["services"].keys())
            for state_name, state_def in (
                resolved["failure"]["remediation"]["states"].items()
            ):
                for action in state_def.get("actions", []):
                    target = action.get("target", "")
                    if target:
                        assert target in res_svcs, (
                            f"{s['id']} seed={seed}: state '{state_name}' "
                            f"action targets '{target}' not in services {res_svcs}"
                        )


def test_loader_with_seed(loader):
    """ScenarioLoader.sample(seed=N) returns randomized scenarios."""
    s0 = loader.sample(scenario_id="kafka_partition_rebalance_storm_001", seed=0)
    s42 = loader.sample(scenario_id="kafka_partition_rebalance_storm_001", seed=42)

    # seed=0 should have original names
    assert "zookeeper" in s0["services"]
    # seed=42 should still have zookeeper (domain-locked) but may have different free services
    assert "zookeeper" in s42["services"]
    # template_vars should be stripped in both
    assert "template_vars" not in s0
    assert "template_vars" not in s42


def test_no_template_vars_in_output(all_scenarios):
    """template_vars key must be stripped from all resolved outputs."""
    for s in all_scenarios:
        for seed in [None, 0, 42]:
            resolved = resolve_scenario(s, seed=seed)
            assert "template_vars" not in resolved, (
                f"{s['id']} seed={seed}: template_vars not stripped"
            )
