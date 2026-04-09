"""Load and sample incident scenarios from JSONL registry files."""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from server.scenario_resolver import resolve_scenario


class ScenarioLoader:
    def __init__(self, registry_path: str = "scenarios/incidents_v3.jsonl"):
        self.scenarios: List[Dict[str, Any]] = []
        self._load(registry_path)
        custom = os.environ.get("OPENENV_CUSTOM_REGISTRY")
        if custom:
            self._load(custom)

    def _load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Scenario registry not found: {path}")
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    scenario = json.loads(line)
                    self._validate(scenario)
                    self.scenarios.append(scenario)

    def _validate(self, scenario: Dict[str, Any]) -> None:
        required = ["id", "title", "difficulty", "duration_minutes",
                     "services", "failure", "log_templates", "metric_templates"]
        missing = [k for k in required if k not in scenario]
        if missing:
            raise ValueError(
                f"Scenario '{scenario.get('id', '?')}' missing keys: {missing}"
            )
        failure = scenario["failure"]
        for k in ["root_service", "root_cause_type", "root_cause_statement"]:
            if k not in failure:
                raise ValueError(
                    f"Scenario '{scenario['id']}' failure missing key: {k}"
                )

    def sample(
        self,
        difficulty: Optional[str] = None,
        scenario_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Sample a scenario, optionally applying parameter randomization.

        Args:
            difficulty: Filter by tier (easy/medium/hard).
            scenario_id: Select a specific scenario by ID.
            seed: Randomization seed for anti-memorization.
                  None or 0 = original scenario (backward compat).
                  Any positive int = deterministic randomized variant.
        """
        if scenario_id:
            matches = [s for s in self.scenarios if s["id"] == scenario_id]
            if not matches:
                raise ValueError(f"Scenario ID not found: {scenario_id}")
            raw = matches[0]
        elif difficulty:
            pool = [s for s in self.scenarios if s["difficulty"] == difficulty]
            if not pool:
                pool = self.scenarios
            raw = random.choice(pool)
        else:
            raw = random.choice(self.scenarios)

        return resolve_scenario(raw, seed=seed)

    def list_difficulties(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self.scenarios:
            d = s["difficulty"]
            counts[d] = counts.get(d, 0) + 1
        return counts
