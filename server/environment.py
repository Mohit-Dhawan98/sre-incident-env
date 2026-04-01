"""SRE Incident Response Environment — MCPEnvironment implementation."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

from server.log_generator import LogGenerator
from server.metric_generator import MetricGenerator
from server.reward import compute_reward
from server.scenario_loader import ScenarioLoader


class SREIncidentEnvironment(MCPEnvironment):
    """RL environment simulating SRE incident investigation.

    Agent discovers and calls MCP tools to investigate a production incident,
    then submits a root-cause diagnosis. Scored via embedding similarity.
    """

    def __init__(self) -> None:
        mcp = FastMCP("sre_incident_env")
        self._register_tools(mcp)
        super().__init__(mcp)

        # Resolve scenario path relative to project root
        scenario_path = Path(__file__).parent.parent / "scenarios" / "incidents.jsonl"
        self.loader = ScenarioLoader(str(scenario_path))

        # Episode state
        self._scenario: Optional[Dict[str, Any]] = None
        self._log_gen: Optional[LogGenerator] = None
        self._metric_gen: Optional[MetricGenerator] = None
        self._difficulty: str = "medium"
        self._query_budget: int = 10
        self._queries_used: int = 0
        self._steps: int = 0
        self._done: bool = False
        self._diagnosis_submitted: bool = False
        self._current_reward: float = 0.0
        self._services_queried: set = set()
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def _register_tools(self, mcp: FastMCP) -> None:
        """Register all SRE investigation tools on the MCP server."""

        @mcp.tool
        def list_services() -> str:
            """List all services involved in the current incident. Free action (no query cost)."""
            if self._scenario is None:
                return json.dumps({"error": "No episode active. Call reset() first."})
            services = list(self._scenario["services"].keys())
            random.shuffle(services)

            # Easy mode: also show the dependency graph
            if self._difficulty == "easy":
                graph = {
                    svc: info.get("upstream", [])
                    for svc, info in self._scenario["services"].items()
                }
                return json.dumps({
                    "services": services,
                    "dependency_graph": graph,
                    "queries_remaining": self._query_budget - self._queries_used,
                })

            return json.dumps({
                "services": services,
                "queries_remaining": self._query_budget - self._queries_used,
            })

        @mcp.tool
        def read_logs(
            service: str,
            window_minutes: int = 5,
            level_filter: Optional[str] = None,
        ) -> str:
            """Read logs for a specific service. Costs 1 query from your budget.

            Args:
                service: Service name to query logs for.
                window_minutes: How many minutes of logs to return (default 5).
                level_filter: Filter by log level: "ERROR", "WARN", "INFO", or None for all.
            """
            if self._done:
                return json.dumps({"error": "Episode is over."})

            self._services_queried.add(service)

            budget_result = self._use_query()
            if budget_result:
                return budget_result

            if self._log_gen is None:
                return json.dumps({"error": "No episode active."})

            logs = self._log_gen.get_logs(service, window_minutes, level_filter)
            if not logs:
                return json.dumps({
                    "logs": [],
                    "message": f"No logs found for service '{service}'",
                    "queries_remaining": self._query_budget - self._queries_used,
                })

            return json.dumps({
                "logs": [l.model_dump() for l in logs],
                "count": len(logs),
                "queries_remaining": self._query_budget - self._queries_used,
            })

        @mcp.tool
        def check_metric(
            service: str,
            metric: str,
            window_minutes: int = 10,
        ) -> str:
            """Check a metric time-series for a specific service. Costs 1 query.

            Args:
                service: Service name to check metrics for.
                metric: Metric name (e.g. "error_rate", "cpu_percent", "query_latency_p99_ms").
                window_minutes: Time window in minutes (default 10).
            """
            if self._done:
                return json.dumps({"error": "Episode is over."})

            self._services_queried.add(service)

            budget_result = self._use_query()
            if budget_result:
                return budget_result

            if self._metric_gen is None:
                return json.dumps({"error": "No episode active."})

            series = self._metric_gen.get_metric(service, metric, window_minutes)
            if not series:
                available = self._metric_gen.list_metrics(service)
                return json.dumps({
                    "metric_series": [],
                    "message": f"Metric '{metric}' not found for '{service}'",
                    "available_metrics": available if available else f"No metrics for '{service}'",
                    "queries_remaining": self._query_budget - self._queries_used,
                })

            return json.dumps({
                "metric": metric,
                "service": service,
                "metric_series": [p.model_dump() for p in series],
                "points": len(series),
                "queries_remaining": self._query_budget - self._queries_used,
            })

        @mcp.tool
        def submit_diagnosis(
            root_cause: str,
            affected_service: str,
            confidence: float = 0.5,
        ) -> str:
            """Submit your root-cause diagnosis. Ends the episode.

            Args:
                root_cause: Your natural-language root cause statement. Be specific: WHO did WHAT causing WHAT.
                affected_service: The service where the root cause originates.
                confidence: Your confidence level 0.0 to 1.0.
            """
            if self._done:
                return json.dumps({"error": "Episode already ended."})
            if self._scenario is None:
                return json.dumps({"error": "No episode active."})

            failure = self._scenario["failure"]
            all_services = set(self._scenario["services"].keys())
            reward = compute_reward(
                submitted_root_cause=root_cause,
                submitted_service=affected_service,
                true_root_cause=failure["root_cause_statement"],
                true_root_service=failure["root_service"],
                steps_used=self._queries_used,
                query_budget=self._query_budget,
                difficulty=self._difficulty,
                confidence=confidence,
                services_queried=self._services_queried,
                all_services=all_services,
            )

            self._done = True
            self._diagnosis_submitted = True
            self._current_reward = reward

            return json.dumps({
                "result": "diagnosis_submitted",
                "reward": reward,
                "done": True,
                "message": f"Diagnosis submitted. Reward: {reward:.4f}",
                "queries_used": self._queries_used,
                "query_budget": self._query_budget,
            })

    def _use_query(self) -> Optional[str]:
        """Deduct a query from the budget. Returns error JSON if exhausted."""
        self._queries_used += 1
        if self._queries_used > self._query_budget:
            self._done = True
            self._current_reward = 0.0
            return json.dumps({
                "error": "Query budget exhausted. Episode ended with reward 0.",
                "done": True,
                "reward": 0.0,
            })
        return None

    @staticmethod
    def _budget_for_difficulty(difficulty: str) -> int:
        return {"easy": 15, "medium": 10, "hard": 7, "expert": 5}.get(difficulty, 10)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        if seed is not None:
            random.seed(seed)

        difficulty = kwargs.get("difficulty", "medium")
        scenario_id = kwargs.get("scenario_id", None)

        self._scenario = self.loader.sample(
            difficulty=difficulty, scenario_id=scenario_id
        )
        self._difficulty = difficulty
        self._query_budget = self._budget_for_difficulty(difficulty)
        self._queries_used = 0
        self._steps = 0
        self._done = False
        self._diagnosis_submitted = False
        self._current_reward = 0.0
        self._services_queried = set()
        self._state = State(
            episode_id=episode_id or str(uuid4()), step_count=0
        )

        self._log_gen = LogGenerator(self._scenario)
        self._metric_gen = MetricGenerator(
            self._scenario, self._log_gen.base_time
        )

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "message": (
                    f"[INCIDENT ALERT] {self._scenario['title']}\n"
                    f"Severity detected. You have {self._query_budget} queries to investigate.\n"
                    f"Use list_services to see available services, then read_logs and check_metric to investigate.\n"
                    f"Submit your diagnosis with submit_diagnosis when ready."
                ),
                "scenario_id": self._scenario["id"],
                "difficulty": difficulty,
                "query_budget": self._query_budget,
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions."""
        return Observation(
            done=self._done,
            reward=self._current_reward,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. Use MCP tools.",
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._steps += 1
        self._state.step_count = self._steps
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        # Overlay our done/reward state onto the observation
        obs.done = self._done
        obs.reward = self._current_reward
        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._steps += 1
        self._state.step_count = self._steps
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        obs.done = self._done
        obs.reward = self._current_reward
        return obs

    @property
    def state(self) -> State:
        return self._state
