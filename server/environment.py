"""SRE Incident Response Environment V2 — Investigation + Remediation State Machine.

Agent investigates a production incident by reading logs and metrics,
then remediates by taking actions that change system state. Wrong actions
make things worse (trap doors). Agent verifies resolution when fixed.
"""

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
from server.state_machine import StateMachine


class SREIncidentEnvironment(MCPEnvironment):
    """RL environment simulating full SRE incident lifecycle.

    Agent discovers services, reads logs/metrics, checks service runbooks,
    applies remediation actions (which change system state), and verifies
    resolution. Wrong remediations make things worse — like a real incident.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        mcp = FastMCP("sre_incident_env")
        self._register_tools(mcp)
        super().__init__(mcp)

        # Use V2 scenarios if available, fall back to V1
        v2_path = Path(__file__).parent.parent / "scenarios" / "incidents_v2.jsonl"
        v1_path = Path(__file__).parent.parent / "scenarios" / "incidents.jsonl"
        scenario_path = v2_path if v2_path.exists() else v1_path
        self.loader = ScenarioLoader(str(scenario_path))

        # Episode state
        self._scenario: Optional[Dict[str, Any]] = None
        self._log_gen: Optional[LogGenerator] = None
        self._metric_gen: Optional[MetricGenerator] = None
        self._state_machine: Optional[StateMachine] = None
        self._difficulty: str = "medium"
        self._query_budget: int = 200
        self._queries_used: int = 0
        self._steps: int = 0
        self._done: bool = False
        self._current_reward: float = 0.0
        self._services_queried: set = set()
        self._tool_call_history: list = []
        # V2.1 maze navigation tracking
        self._discovered_services: set = set()          # get_service_info called on these
        self._remediation_count: int = 0                 # total remediation tool calls
        self._execute_runbook_count: int = 0             # execute_runbook calls specifically
        self._discovered_before_action: int = 0          # execute_runbook where get_service_info was called first
        self._observation_after_fix: int = 0             # read_logs right after a remediation
        self._last_was_remediation: bool = False         # for observe-after-fix tracking
        self._unique_remediation_keys: set = set()       # unique (tool, target, action) combos
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def _has_remediation(self) -> bool:
        """Check if current scenario has V2 remediation data."""
        if not self._scenario:
            return False
        return "remediation" in self._scenario.get("failure", {})

    def _register_tools(self, mcp: FastMCP) -> None:
        """Register all SRE tools on the MCP server."""

        # ─── INVESTIGATION TOOLS ───────────────────────────────────

        @mcp.tool
        def list_services() -> str:
            """List all services involved in the current incident. Free action (no query cost).

            Returns service names and dependency graph (on easy difficulty).
            Call this first to understand the topology.
            """
            if self._scenario is None:
                return json.dumps({"error": "No episode active. Call reset() first."})
            if self._done:
                return json.dumps({"error": "Episode is over."})
            services = list(self._scenario["services"].keys())
            random.shuffle(services)

            if self._difficulty == "easy":
                graph = {
                    svc: info.get("upstream", [])
                    for svc, info in self._scenario["services"].items()
                }
                return json.dumps({"services": services, "dependency_graph": graph})

            return json.dumps({"services": services})

        @mcp.tool
        def read_logs(
            service: str,
            window_minutes: int = 5,
            level_filter: Optional[str] = None,
        ) -> str:
            """Read logs for a specific service. Costs 1 query.

            After remediation actions, new logs may appear reflecting the changed system state.
            Always read_logs after remediating to observe the outcome.

            Args:
                service: Service name to query logs for.
                window_minutes: How many minutes of logs to return (default 5).
                level_filter: Filter by log level: "ERROR", "WARN", "INFO", or None for all.
            """
            if self._done:
                return json.dumps({"error": "Episode is over."})
            budget_result = self._use_query()
            if budget_result:
                return budget_result

            self._services_queried.add(service)
            self._tool_call_history.append(("read_logs", service))
            if self._last_was_remediation:
                self._observation_after_fix += 1
            self._last_was_remediation = False

            if self._log_gen is None:
                return json.dumps({"error": "No episode active."})

            logs = self._log_gen.get_logs(service, window_minutes, level_filter)

            # Merge post-remediation overlay logs
            if self._state_machine:
                overlay = [
                    l for l in self._state_machine.overlay_logs
                    if l.service.lower() == service.lower()
                ]
                if level_filter:
                    overlay = [l for l in overlay if l.level.upper() == level_filter.upper()]
                logs = list(logs) + overlay
                logs.sort(key=lambda l: l.timestamp)

            if not logs:
                return json.dumps({
                    "logs": [],
                    "message": f"No logs found for service '{service}'",
                })

            return json.dumps({
                "logs": [l.model_dump() for l in logs],
                "count": len(logs),
            })

        @mcp.tool
        def check_metric(
            service: str,
            metric: str,
            window_minutes: int = 10,
        ) -> str:
            """Check a metric time-series for a specific service. Costs 1 query.

            After remediation actions, metrics may change reflecting the new system state.

            Args:
                service: Service name to check metrics for.
                metric: Metric name (e.g. "error_rate", "cpu_percent", "latency_p99_ms").
                window_minutes: Time window in minutes (default 10).
            """
            if self._done:
                return json.dumps({"error": "Episode is over."})
            budget_result = self._use_query()
            if budget_result:
                return budget_result

            self._services_queried.add(service)
            self._tool_call_history.append(("check_metric", service))
            if self._last_was_remediation:
                self._observation_after_fix += 1
            self._last_was_remediation = False

            if self._metric_gen is None:
                return json.dumps({"error": "No episode active."})

            series = self._metric_gen.get_metric(service, metric, window_minutes)
            if not series:
                available = self._metric_gen.list_metrics(service)
                return json.dumps({
                    "metric_series": [],
                    "message": f"Metric '{metric}' not found for '{service}'",
                    "available_metrics": available if available else f"No metrics for '{service}'",
                })

            return json.dumps({
                "metric": metric,
                "service": service,
                "metric_series": [p.model_dump() for p in series],
                "points": len(series),
            })

        @mcp.tool
        def get_service_info(service: str) -> str:
            """Get the service catalog / runbook entry for a service. Costs 1 query.

            Returns available maintenance actions, configurable parameters, recent
            deployments, and health checks. Use this to discover what actions you
            can take on a service before attempting remediation.

            Args:
                service: Service name to look up in the catalog.
            """
            if self._done:
                return json.dumps({"error": "Episode is over."})
            budget_result = self._use_query()
            if budget_result:
                return budget_result

            self._services_queried.add(service)
            self._tool_call_history.append(("get_service_info", service))
            self._discovered_services.add(service.lower())
            self._last_was_remediation = False

            if not self._state_machine:
                return json.dumps({"error": "No episode active."})

            info = self._state_machine.get_service_info(service)
            if info is None:
                return json.dumps({
                    "error": f"Service '{service}' not found in the service catalog.",
                })

            return json.dumps({
                "service": service,
                "description": info.get("description", ""),
                "available_actions": info.get("available_actions", []),
                "configurable_params": info.get("configurable_params", []),
                "recent_deploys": info.get("recent_deploys", []),
                "health_checks": info.get("health_checks", []),
            })

        # ─── PLATFORM REMEDIATION TOOLS ────────────────────────────

        @mcp.tool
        def restart_service(service: str) -> str:
            """Restart a service process. Clears runtime state (memory, connections, caches). Costs 1 query.

            Use when the problem is transient runtime state: leaked connections,
            exhausted pools, filled memory. Does NOT help with config drift,
            expired certificates, missing indexes, or kernel-level issues —
            the service reloads the same broken config on restart.

            Args:
                service: Service name to restart.
            """
            return self._handle_remediation("restart_service", service)

        @mcp.tool
        def rollback_deploy(service: str) -> str:
            """Revert a service to its previous deployment version. Costs 1 query.

            Use when a recent deployment introduced the bug. Check
            get_service_info for recent_deploys to see if a deploy happened.
            Only works if there is a previous version to roll back to.

            Args:
                service: Service name to roll back.
            """
            return self._handle_remediation("rollback_deploy", service)

        @mcp.tool
        def scale_replicas(service: str, count: int = 3) -> str:
            """Scale a service horizontally by changing replica count. Costs 1 query.

            Use when the system needs more capacity (traffic spike, thundering herd).
            WARNING: Scaling a leaking/broken service just multiplies the problem —
            more instances = more leaked connections, more OOM kills, etc.

            Args:
                service: Service name to scale.
                count: Target number of replicas (e.g. 3, 5, 10).
            """
            return self._handle_remediation(
                "scale_replicas", service, {"count": count}
            )

        # ─── V1 BACKWARD COMPAT ────────────────────────────────────

        @mcp.tool
        def submit_diagnosis(
            affected_service: str,
            failure_type: str,
            root_cause: str,
            confidence: float = 0.5,
            causal_chain: Optional[str] = None,
        ) -> str:
            """Submit your root-cause diagnosis. Ends the episode.

            Args:
                affected_service: The service where the root cause ORIGINATES.
                failure_type: Category of failure (e.g. connection_leak, config_drift, etc.)
                root_cause: "[service] [mechanism] caused [downstream effect]".
                confidence: Your confidence 0.0 to 1.0.
                causal_chain: Comma-separated services from root to symptom.
            """
            if self._done:
                return json.dumps({"error": "Episode already ended."})
            if self._scenario is None:
                return json.dumps({"error": "No episode active."})

            chain_list = []
            if causal_chain:
                chain_list = [s.strip() for s in causal_chain.split(",") if s.strip()]

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
                submitted_failure_type=failure_type,
                true_failure_type=failure["root_cause_type"],
                submitted_chain=chain_list,
                services_graph=self._scenario["services"],
                tool_call_history=self._tool_call_history,
                true_causal_chain=failure.get("causal_chain", []),
                optimal_queries=failure.get("optimal_queries", 10),
                explanation_keywords=failure.get("explanation_keywords", []),
            )
            self._done = True
            self._current_reward = reward
            return json.dumps({
                "result": "diagnosis_submitted",
                "reward": reward,
                "done": True,
                "message": f"Diagnosis submitted. Reward: {reward:.4f}",
            })

        # ─── APPLICATION REMEDIATION (META TOOL) ───────────────────

        @mcp.tool
        def execute_runbook(
            service: str,
            action: str,
            params: Optional[str] = None,
        ) -> str:
            """Execute a service-specific maintenance action from the runbook. Costs 1 query.

            Use get_service_info first to discover available actions for a service.
            This tool handles all service-specific operations including config changes,
            cache operations, certificate management, database maintenance, and more.

            Examples:
              execute_runbook("session-db", "update_config", '{"key": "connection_timeout", "value": "30"}')
              execute_runbook("billing-svc", "renew_certificates")
              execute_runbook("etcd-cluster", "trigger_compaction")
              execute_runbook("cache-cluster", "flush_all")
              execute_runbook("analytics-db", "create_index", '{"name": "idx_events_user_created_at"}')

            Args:
                service: Service name to run the action on.
                action: Action name from the service's available_actions list.
                params: Optional JSON string of action parameters (e.g. '{"key": "timeout", "value": "30"}').
            """
            parsed_params = {}
            if params:
                try:
                    parsed_params = json.loads(params) if isinstance(params, str) else params
                except (json.JSONDecodeError, TypeError):
                    return json.dumps({
                        "error": f"Invalid params format. Expected JSON string, got: {params}",
                    })

            # For execute_runbook, the action name is part of matching
            parsed_params["_action"] = action
            return self._handle_remediation("execute_runbook", service, parsed_params)

        # ─── TERMINAL TOOL ─────────────────────────────────────────

        @mcp.tool
        def verify_resolution(
            affected_service: str,
            failure_type: str,
            root_cause: str,
            confidence: float = 0.5,
            causal_chain: Optional[str] = None,
        ) -> str:
            """Verify the system is healthy and submit your diagnosis. Ends the episode.

            Call this after you have remediated the incident. The system will check
            if your fix actually worked AND grade your diagnosis.

            You can call this even if the system isn't fixed — you'll get partial
            credit for correct diagnosis but zero for remediation.

            Args:
                affected_service: The service where the root cause ORIGINATES.
                failure_type: Category of failure (e.g. connection_leak, config_drift, cert_expiry, cache_stampede, etc.)
                root_cause: Explanation: "[service] [mechanism] caused [downstream effect]".
                confidence: Your confidence 0.0 to 1.0.
                causal_chain: Comma-separated services from root cause to visible symptom (e.g. "svc-a,svc-b,svc-c").
            """
            if self._done:
                return json.dumps({"error": "Episode already ended."})
            if self._scenario is None:
                return json.dumps({"error": "No episode active."})

            chain_list = []
            if causal_chain:
                chain_list = [s.strip() for s in causal_chain.split(",") if s.strip()]

            failure = self._scenario["failure"]
            all_services = set(self._scenario["services"].keys())

            # Determine system health from state machine
            system_healthy = False
            remediation_attempts = 0
            harm_count = 0
            correct_fix_used = False
            first_try_correct = False

            if self._state_machine:
                system_healthy = self._state_machine.is_resolved()
                remediation_attempts = self._state_machine.remediation_attempts
                harm_count = len(self._state_machine.harm_events)
                # Check if any correct action was used
                for tool, target, params, outcome in self._state_machine.remediation_history:
                    if outcome == "recovery":
                        correct_fix_used = True
                        if self._state_machine.remediation_history.index((tool, target, params, outcome)) == 0:
                            first_try_correct = True
                        break

            # Get optimal_steps from remediation data
            rem_data = failure.get("remediation", {})
            optimal_steps = rem_data.get("optimal_steps", 2)

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
                submitted_failure_type=failure_type,
                true_failure_type=failure["root_cause_type"],
                submitted_chain=chain_list,
                services_graph=self._scenario["services"],
                tool_call_history=self._tool_call_history,
                true_causal_chain=failure.get("causal_chain", []),
                optimal_queries=failure.get("optimal_queries", 10),
                explanation_keywords=failure.get("explanation_keywords", []),
                # V2.1 maze navigation params
                system_healthy=system_healthy,
                remediation_attempts=remediation_attempts,
                harm_count=harm_count,
                correct_fix_used=correct_fix_used,
                first_try_correct=first_try_correct,
                optimal_steps=optimal_steps,
                remediation_count=self._remediation_count,
                observation_after_fix=self._observation_after_fix,
                discovered_before_action=self._discovered_before_action,
                execute_runbook_count=self._execute_runbook_count,
                unique_remediation_count=len(self._unique_remediation_keys),
            )

            self._done = True
            self._current_reward = reward

            system_state = "unknown"
            if self._state_machine:
                system_state = self._state_machine.system_state

            return json.dumps({
                "result": "resolution_verified",
                "system_state": system_state,
                "system_healthy": system_healthy,
                "reward": reward,
                "done": True,
                "message": f"Resolution verified. System: {system_state}. Reward: {reward:.4f}",
            })

    # ─── HELPERS ───────────────────────────────────────────────

    def _handle_remediation(
        self, tool: str, service: str, params: Optional[Dict] = None
    ) -> str:
        """Common handler for all remediation tools."""
        if self._done:
            return json.dumps({"error": "Episode is over."})

        budget_result = self._use_query()
        if budget_result:
            return budget_result

        self._tool_call_history.append((tool, service))

        # Track maze navigation signals
        self._remediation_count += 1
        self._last_was_remediation = True

        action_name = ""
        if params and "_action" in params:
            action_name = params["_action"]

        # Track unique remediation calls
        rem_key = f"{tool}:{service}:{action_name}".lower()
        self._unique_remediation_keys.add(rem_key)

        # Track discovery before execute_runbook
        if tool == "execute_runbook":
            self._execute_runbook_count += 1
            if service.lower() in self._discovered_services:
                self._discovered_before_action += 1

        if not self._state_machine:
            return json.dumps({"error": "No episode active."})

        if not self._has_remediation():
            return json.dumps({
                "error": "This scenario does not support remediation actions. Use submit_diagnosis instead.",
            })

        outcome = self._state_machine.process_remediation(tool, service, params)

        return json.dumps({
            "action": tool,
            "target": service,
            "outcome": outcome.outcome,
            "system_health": outcome.post_state,
            "message": outcome.message,
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
        return 200

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
        self._current_reward = 0.0
        self._services_queried = set()
        self._tool_call_history = []
        # Reset V2.1 maze tracking
        self._discovered_services = set()
        self._remediation_count = 0
        self._execute_runbook_count = 0
        self._discovered_before_action = 0
        self._observation_after_fix = 0
        self._last_was_remediation = False
        self._unique_remediation_keys = set()
        self._state = State(
            episode_id=episode_id or str(uuid4()), step_count=0
        )

        self._log_gen = LogGenerator(self._scenario)
        self._metric_gen = MetricGenerator(
            self._scenario, self._log_gen.base_time
        )

        # Initialize state machine for V2 scenarios
        self._state_machine = StateMachine(
            self._scenario, self._log_gen.base_time
        )

        # Build alert message
        terminal_tool = "verify_resolution" if self._has_remediation() else "submit_diagnosis"
        alert = (
            f"[INCIDENT ALERT] {self._scenario['title']}\n"
            f"Severity detected. Investigate using the available tools.\n"
            f"Use list_services to see the topology, then read_logs and check_metric to investigate.\n"
        )
        if self._has_remediation():
            alert += (
                f"Use get_service_info to discover available actions on each service.\n"
                f"Apply fixes with restart_service, rollback_deploy, scale_replicas, or execute_runbook.\n"
                f"Call verify_resolution when the system is healthy."
            )
        else:
            alert += f"Submit your diagnosis with submit_diagnosis when ready."

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "message": alert,
                "scenario_id": self._scenario["id"],
                "difficulty": difficulty,
            },
        )

    def _step_impl(
        self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any,
    ) -> Observation:
        return Observation(
            done=self._done,
            reward=self._current_reward,
            metadata={"error": f"Unknown action type: {type(action).__name__}. Use MCP tools."},
        )

    def step(
        self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any,
    ) -> Observation:
        self._steps += 1
        self._state.step_count = self._steps
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        obs.done = self._done
        obs.reward = self._current_reward
        return obs

    async def step_async(
        self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any,
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
