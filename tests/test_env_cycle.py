"""Test full episode lifecycle: reset → step × N → done."""

import json
import pytest
from server.environment import SREIncidentEnvironment
from openenv.core.env_server.mcp_types import CallToolAction


@pytest.fixture
def env():
    return SREIncidentEnvironment()


def _tool_data(obs) -> dict:
    """Extract tool result data from observation."""
    if hasattr(obs, 'result') and obs.result and hasattr(obs.result, 'data'):
        return json.loads(obs.result.data)
    return {}


def test_full_episode_easy(env):
    obs = env.reset(seed=42, difficulty="easy")
    assert not obs.done
    assert obs.reward == 0.0
    assert obs.metadata["query_budget"] == 15

    # List services (free)
    obs = env.step(CallToolAction(tool_name="list_services", arguments={}))
    data = _tool_data(obs)
    assert "services" in data
    assert len(data["services"]) >= 3
    # Easy mode shows dependency graph
    assert "dependency_graph" in data

    # Read logs
    svc = data["services"][0]
    obs = env.step(CallToolAction(
        tool_name="read_logs",
        arguments={"service": svc, "window_minutes": 15},
    ))
    assert not obs.done

    # Submit diagnosis
    obs = env.step(CallToolAction(
        tool_name="submit_diagnosis",
        arguments={
            "affected_service": svc,
            "failure_type": "cache_node_failure",
            "root_cause": "cache node failure",
            "confidence": 0.7,
        },
    ))
    assert obs.done
    assert 0.0 <= obs.reward <= 1.0


def test_full_episode_medium(env):
    obs = env.reset(seed=123, difficulty="medium")
    assert obs.metadata["query_budget"] == 10

    obs = env.step(CallToolAction(tool_name="list_services", arguments={}))
    data = _tool_data(obs)
    # Medium mode: no dependency graph
    assert "dependency_graph" not in data

    svc = data["services"][0]
    obs = env.step(CallToolAction(
        tool_name="read_logs",
        arguments={"service": svc, "window_minutes": 10, "level_filter": "ERROR"},
    ))
    log_data = _tool_data(obs)
    assert "logs" in log_data or "message" in log_data


def test_check_metric(env):
    env.reset(seed=42, difficulty="easy")
    obs = env.step(CallToolAction(tool_name="list_services", arguments={}))
    services = _tool_data(obs)["services"]

    obs = env.step(CallToolAction(
        tool_name="check_metric",
        arguments={"service": services[0], "metric": "error_rate"},
    ))
    data = _tool_data(obs)
    assert "metric_series" in data or "message" in data


def test_invalid_service_returns_empty(env):
    env.reset(seed=42, difficulty="medium")
    obs = env.step(CallToolAction(
        tool_name="read_logs",
        arguments={"service": "nonexistent-service", "window_minutes": 5},
    ))
    data = _tool_data(obs)
    assert data.get("logs") == [] or "No logs found" in data.get("message", "")


def test_budget_exhaustion(env):
    env.reset(seed=42, difficulty="expert")  # budget = 5
    for i in range(5):
        obs = env.step(CallToolAction(
            tool_name="read_logs",
            arguments={"service": "any", "window_minutes": 5},
        ))
    # 6th query should exhaust budget
    obs = env.step(CallToolAction(
        tool_name="read_logs",
        arguments={"service": "any", "window_minutes": 5},
    ))
    assert obs.done
    assert obs.reward == 0.0


def test_step_after_done_raises(env):
    env.reset(seed=42, difficulty="easy")
    env.step(CallToolAction(
        tool_name="submit_diagnosis",
        arguments={"affected_service": "test", "failure_type": "other", "root_cause": "test", "confidence": 0.5},
    ))
    # Next tool call should get error response (episode done)
    obs = env.step(CallToolAction(
        tool_name="list_services",
        arguments={},
    ))
    data = _tool_data(obs)
    assert "error" in data.get("error", "") or obs.done
