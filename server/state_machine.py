"""State machine for SRE incident remediation — Graph-based traversal.

Each scenario defines a state GRAPH where:
- Each state has its own set of valid actions
- Actions transition to other states (forward, sideways, backward)
- The agent navigates the maze to reach a resolved state
- Post-logs/metrics are generated per transition

Supports two data formats:
1. Graph format (V2.1): states dict with per-state actions
2. Flat format (V2.0): correct_actions/partial_actions/trap_actions (auto-converted)
"""

import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from models import LogEntry


class RemediationOutcome:
    """Result of a remediation action."""

    def __init__(
        self,
        outcome: str,  # "recovery", "progress", "worsened", "no_effect"
        message: str,
        post_state: str,
        post_logs: Optional[List[Dict]] = None,
        post_metrics: Optional[Dict] = None,
    ):
        self.outcome = outcome
        self.message = message
        self.post_state = post_state
        self.post_logs = post_logs or []
        self.post_metrics = post_metrics or {}


class StateMachine:
    """Graph-based state machine for incident remediation.

    Each state has its own action table. Actions transition between states.
    The agent navigates the graph to reach a resolved state.
    """

    def __init__(self, scenario: Dict[str, Any], base_time: datetime):
        self.scenario = scenario
        self.base_time = base_time
        self.duration_minutes = scenario.get("duration_minutes", 15)

        # Load remediation data
        failure = scenario.get("failure", {})
        remediation = failure.get("remediation", {})
        self.service_info = scenario.get("service_info", {})

        # Detect format and load states
        if "states" in remediation:
            # V2.1 graph format
            self.states = remediation["states"]
            self.initial_state = remediation.get("initial_state", "broken")
            self.resolved_states = set(remediation.get("resolved_states", ["healthy"]))
            self.optimal_steps = remediation.get("optimal_steps", 2)
        else:
            # V2.0 flat format — convert to graph
            self.states = self._convert_flat_to_graph(remediation)
            self.initial_state = "broken"
            self.resolved_states = {"healthy"}
            self.optimal_steps = remediation.get("optimal_steps", 1)

        # Current state
        self.system_state = self.initial_state

        # Tracking
        self.remediation_history: List[Tuple[str, str, Dict, str]] = []
        self.harm_events: List[Tuple[str, str]] = []
        self.remediation_attempts = 0
        self.overlay_logs: List[LogEntry] = []
        self.overlay_metrics: Dict[str, Dict] = {}
        self._action_time_offset = 0  # seconds after base_time + duration for overlay timestamps

    def _convert_flat_to_graph(self, remediation: Dict) -> Dict:
        """Convert V2.0 flat format to V2.1 graph format.

        Flat format has: correct_actions, partial_actions, trap_actions, default_response
        Converts to a 2-state graph: broken → healthy (with traps → critical)
        """
        if not remediation:
            return {
                "broken": {
                    "actions": [],
                    "default": {"next_state": "broken", "outcome": "no_effect",
                               "message": "No remediation data for this scenario."}
                },
                "healthy": {
                    "is_resolved": True, "actions": [],
                    "default": {"next_state": "healthy", "outcome": "no_effect",
                               "message": "System is healthy."}
                }
            }

        broken_actions = []

        for ca in remediation.get("correct_actions", []):
            broken_actions.append({
                "tool": ca["tool"], "target": ca["target"],
                "params": ca.get("params", {}),
                "next_state": "healthy", "outcome": "recovery",
                "message": ca.get("message", "System fixed."),
                "post_logs": ca.get("post_logs", []),
                "post_metrics": ca.get("post_metrics", {})
            })

        for pa in remediation.get("partial_actions", []):
            broken_actions.append({
                "tool": pa["tool"], "target": pa["target"],
                "params": pa.get("params", {}),
                "next_state": pa.get("post_state", "degraded"),
                "outcome": "progress" if pa.get("post_state") == "healthy" else "no_effect",
                "message": pa.get("message", "Partial effect."),
                "post_logs": pa.get("post_logs", []),
                "post_metrics": pa.get("post_metrics", {})
            })

        for ta in remediation.get("trap_actions", []):
            broken_actions.append({
                "tool": ta["tool"], "target": ta["target"],
                "params": ta.get("params", {}),
                "next_state": ta.get("post_state", "critical"),
                "outcome": "worsened",
                "message": ta.get("message", "Things got worse."),
                "post_logs": ta.get("post_logs", []),
                "post_metrics": ta.get("post_metrics", {})
            })

        default = remediation.get("default_response", {})

        return {
            "broken": {
                "actions": broken_actions,
                "default": {
                    "next_state": default.get("post_state", "broken"),
                    "outcome": "no_effect",
                    "message": default.get("message", "No observable effect.")
                }
            },
            "degraded": {
                "actions": broken_actions,  # Same actions available from degraded
                "default": {
                    "next_state": "degraded", "outcome": "no_effect",
                    "message": "System degraded. " + default.get("message", "")
                }
            },
            "critical": {
                "actions": broken_actions,  # Can still recover from critical
                "default": {
                    "next_state": "critical", "outcome": "no_effect",
                    "message": "System critical. " + default.get("message", "")
                }
            },
            "healthy": {
                "is_resolved": True, "actions": [],
                "default": {
                    "next_state": "healthy", "outcome": "no_effect",
                    "message": "System is healthy. No further action needed."
                }
            }
        }

    def _extract_action_name(self, params: Dict) -> str:
        return params.get("_action", "").lower()

    def _action_matches(self, action_def: Dict, tool: str, target: str, params: Dict) -> bool:
        """Check if an action definition matches the agent's call."""
        if action_def["tool"].lower() != tool.lower():
            return False
        if action_def["target"].lower() != target.lower():
            return False

        golden_params = action_def.get("params", {})

        if tool == "execute_runbook":
            agent_action = self._extract_action_name(params)
            golden_action = self._extract_action_name(golden_params)

            if not agent_action or not golden_action:
                return False
            if agent_action != golden_action:
                return False

            # Check required params beyond _action
            required = {k: v for k, v in golden_params.items()
                       if k != "_action" and v is not None}
            if required:
                for key, value in required.items():
                    provided = params.get(key)
                    if provided is None:
                        return False
                    if str(provided).lower() != str(value).lower():
                        return False

        return True

    def get_service_info(self, service: str) -> Optional[Dict]:
        """Get service catalog entry."""
        for svc_name, info in self.service_info.items():
            if svc_name.lower() == service.lower():
                return info
        return None

    def process_remediation(
        self, tool: str, target: str, params: Optional[Dict] = None
    ) -> RemediationOutcome:
        """Process a remediation action in the current state.

        Looks up action in CURRENT STATE's action table.
        Returns outcome and transitions to new state.
        """
        self.remediation_attempts += 1
        params = params or {}

        # Check if target service exists
        all_services = {s.lower() for s in self.scenario.get("services", {}).keys()}
        if target.lower() not in all_services:
            return RemediationOutcome(
                outcome="no_effect",
                message=f"Service '{target}' not found in the current topology.",
                post_state=self.system_state,
            )

        # Get current state definition
        state_def = self.states.get(self.system_state)
        if not state_def:
            return RemediationOutcome(
                outcome="no_effect",
                message=f"Unknown system state.",
                post_state=self.system_state,
            )

        # Check if already resolved
        if state_def.get("is_resolved"):
            return RemediationOutcome(
                outcome="no_effect",
                message="System is already healthy. No further action needed.",
                post_state=self.system_state,
            )

        # Find matching action in current state
        for action_def in state_def.get("actions", []):
            if self._action_matches(action_def, tool, target, params):
                outcome_type = action_def.get("outcome", "no_effect")
                next_state = action_def.get("next_state", self.system_state)

                # Track
                self.system_state = next_state
                self.remediation_history.append((tool, target, params, outcome_type))
                if outcome_type == "worsened":
                    self.harm_events.append((tool, action_def.get("message", "")))

                # Apply post-logs/metrics
                self._apply_action_overlays(action_def)

                return RemediationOutcome(
                    outcome=outcome_type,
                    message=action_def.get("message", "Action completed."),
                    post_state=next_state,
                    post_logs=action_def.get("post_logs", []),
                    post_metrics=action_def.get("post_metrics", {}),
                )

        # No match — use default for this state
        default = state_def.get("default", {})
        default_state = default.get("next_state", self.system_state)
        self.system_state = default_state
        self.remediation_history.append((tool, target, params, "no_effect"))

        # For execute_runbook with unknown action, give helpful info
        if tool == "execute_runbook":
            agent_action = self._extract_action_name(params)
            svc_info = self.get_service_info(target)
            if svc_info and agent_action:
                available = svc_info.get("available_actions", [])
                if agent_action not in [a.lower() for a in available]:
                    return RemediationOutcome(
                        outcome="no_effect",
                        message=f"Action '{agent_action}' not available on {target}. Check get_service_info for valid actions.",
                        post_state=default_state,
                    )

        return RemediationOutcome(
            outcome="no_effect",
            message=default.get("message", "No observable effect on the incident."),
            post_state=default_state,
        )

    def _apply_action_overlays(self, action_def: Dict) -> None:
        """Generate overlay logs/metrics from an action's post-data."""
        self._action_time_offset += 5  # Each action advances time slightly
        action_time = self.base_time + timedelta(
            minutes=self.duration_minutes,
            seconds=self._action_time_offset
        )

        for log_def in action_def.get("post_logs", []):
            offset = log_def.get("offset_after_action_seconds", 5)
            ts = action_time + timedelta(seconds=offset)

            template = log_def.get("template", "")
            log_vars = log_def.get("log_vars", {})
            values = {"ts": ts.strftime("%Y-%m-%d %H:%M:%S")}
            for var_name, var_def in log_vars.items():
                if isinstance(var_def, dict):
                    if "min" in var_def and "max" in var_def:
                        values[var_name] = random.randint(
                            int(var_def["min"]), int(var_def["max"])
                        )
                    elif "choices" in var_def:
                        values[var_name] = random.choice(var_def["choices"])
                else:
                    values[var_name] = var_def

            try:
                message = template.format(**values)
            except (KeyError, IndexError):
                message = template

            self.overlay_logs.append(LogEntry(
                timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
                service=log_def.get("service", "unknown"),
                level=log_def.get("level", "INFO"),
                message=message,
            ))

        for svc, metrics in action_def.get("post_metrics", {}).items():
            if svc not in self.overlay_metrics:
                self.overlay_metrics[svc] = {}
            self.overlay_metrics[svc].update(metrics)

    def is_resolved(self) -> bool:
        """Check if the system has reached a resolved state."""
        state_def = self.states.get(self.system_state, {})
        if state_def.get("is_resolved"):
            return True
        return self.system_state in self.resolved_states
