"""Test scenario loading and validation."""

import json
import pytest
import tempfile
from pathlib import Path
from server.scenario_loader import ScenarioLoader


def test_load_builtin():
    loader = ScenarioLoader("scenarios/incidents.jsonl")
    assert len(loader.scenarios) >= 5


def test_sample_by_difficulty():
    loader = ScenarioLoader("scenarios/incidents.jsonl")
    s = loader.sample(difficulty="easy")
    assert s["difficulty"] == "easy"


def test_sample_by_id():
    loader = ScenarioLoader("scenarios/incidents.jsonl")
    s = loader.sample(scenario_id=loader.scenarios[0]["id"])
    assert s["id"] == loader.scenarios[0]["id"]


def test_sample_unknown_id():
    loader = ScenarioLoader("scenarios/incidents.jsonl")
    with pytest.raises(ValueError, match="not found"):
        loader.sample(scenario_id="nonexistent_id")


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        ScenarioLoader("/tmp/does_not_exist.jsonl")


def test_custom_registry():
    scenario = {
        "id": "test_001",
        "title": "Test",
        "difficulty": "easy",
        "duration_minutes": 10,
        "services": {"svc-a": {"upstream": []}},
        "failure": {
            "root_service": "svc-a",
            "root_cause_type": "test",
            "root_cause_statement": "test cause",
        },
        "log_templates": [],
        "metric_templates": {},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(scenario) + "\n")
        f.flush()
        loader = ScenarioLoader(f.name)
        assert len(loader.scenarios) == 1
        assert loader.scenarios[0]["id"] == "test_001"


def test_list_difficulties():
    loader = ScenarioLoader("scenarios/incidents.jsonl")
    diffs = loader.list_difficulties()
    assert "easy" in diffs
    assert all(v > 0 for v in diffs.values())
