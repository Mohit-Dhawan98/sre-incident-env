"""State machine for SRE incident remediation.

Handles state transitions when agents take remediation actions.
Looks up (tool, target, params) in scenario remediation data and returns outcomes.

States: "broken" → "degraded" | "critical" | "recovering" | "healthy"

Every remediation action either:
- Fixes the system (correct_action → "healthy")
- Provides temporary relief (partial_action → "degraded")
- Makes things worse (trap_action → "critical")
- Has no effect (default → no state change)
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
        outcome: str,  # "recovery", "partial", "worsened", "no_effect"
        message: str,
        post_state: str,  # "healthy", "degraded", "critical", "broken"
        post_logs: Optional[List[Dict]] = None,
        post_metrics: Optional[Dict] = None,
    ):
        self.outcome = outcome
        self.message = message
        self.post_state = post_state
        self.post_logs = post_logs or []
        self.post_metrics = post_metrics or {}


class StateMachine:
    """Manages system state transitions during incident remediation.

    Initialized with a scenario's remediation data. Processes remediation
    actions and returns outcomes with new logs/metrics to overlay.
    """

    def __init__(self, scenario: Dict[str, Any], base_time: datetime):
        self.scenario = scenario
        self.base_time = base_time
        self.duration_minutes = scenario.get("duration_minutes", 15)

        # Current system state
        self.system_state = "broken"

        # Remediation tracking
        self.remediation_history: List[Tuple[str, str, Dict, str]] = []
        self.harm_events: List[Tuple[str, str]] = []
        self.remediation_attempts = 0

        # Post-remediation overlay data
        self.overlay_logs: List[LogEntry] = []
        self.overlay_metrics: Dict[str, Dict] = {}

        # Load remediation data from scenario
        failure = scenario.get("failure", {})
        self.remediation_data = failure.get("remediation", {})
        self.service_info = scenario.get("service_info", {})

        # Index actions as lists (multiple actions can share tool+target)
        self._correct_actions: Dict[str, List[Dict]] = {}
        self._partial_actions: Dict[str, List[Dict]] = {}
        self._trap_actions: Dict[str, List[Dict]] = {}

        for action in self.remediation_data.get("correct_actions", []):
            key = self._action_key(action["tool"], action["target"])
            self._correct_actions.setdefault(key, []).append(action)

        for action in self.remediation_data.get("partial_actions", []):
            key = self._action_key(action["tool"], action["target"])
            self._partial_actions.setdefault(key, []).append(action)

        for action in self.remediation_data.get("trap_actions", []):
            key = self._action_key(action["tool"], action["target"])
            self._trap_actions.setdefault(key, []).append(action)

    def _action_key(self, tool: str, target: str) -> str:
        """Create lookup key from tool name and target service."""
        return f"{tool.lower()}:{target.lower()}"

    def get_service_info(self, service: str) -> Optional[Dict]:
        """Get service catalog entry for a service."""
        # Case-insensitive lookup
        for svc_name, info in self.service_info.items():
            if svc_name.lower() == service.lower():
                return info
        return None

    def _extract_action_name(self, params: Dict) -> str:
        """Extract the runbook action name from params."""
        return params.get("_action", "").lower()

    def _match_action_in_list(
        self, actions: List[Dict], tool: str, params: Dict
    ) -> Optional[Dict]:
        """Find matching action from a list.

        For execute_runbook: matches on _action name, then checks required golden params.
        For platform tools (restart/rollback/scale): matches first entry (no action name needed).

        Golden params define what's REQUIRED. If golden params is {} (empty),
        the action name alone is sufficient — no param checking needed.
        If golden defines specific key/value pairs, agent must provide those exact values.
        """
        agent_action = self._extract_action_name(params)

        for action_def in actions:
            golden_params = action_def.get("params", {})
            golden_action = golden_params.get("_action", "").lower()

            if tool == "execute_runbook":
                # Must match action name
                if not agent_action or not golden_action:
                    continue
                if agent_action != golden_action:
                    continue
                # Action name matches — check required params from golden
                # Only params explicitly defined in golden (besides _action) must match
                required = {k: v for k, v in golden_params.items() if k != "_action" and v is not None}
                if not required:
                    # No params required beyond action name — match!
                    return action_def
                if self._params_match(required, params):
                    return action_def
            else:
                # Platform tools: first entry matches
                return action_def

        return None

    def process_remediation(
        self, tool: str, target: str, params: Optional[Dict] = None
    ) -> RemediationOutcome:
        """Process a remediation action and return the outcome.

        Matching strategy:
        - For platform tools (restart/rollback/scale): match on tool + target service
        - For execute_runbook: match on tool + target + action name (from params._action)
        - Params beyond action name are matched loosely (golden must be subset of provided)
        """
        self.remediation_attempts += 1
        params = params or {}

        # Check if target service exists in scenario
        all_services = {s.lower() for s in self.scenario.get("services", {}).keys()}
        if target.lower() not in all_services:
            return RemediationOutcome(
                outcome="no_effect",
                message=f"Service '{target}' not found in the current topology.",
                post_state=self.system_state,
            )

        key = self._action_key(tool, target)

        # 1. Check correct actions
        if key in self._correct_actions:
            matched = self._match_action_in_list(self._correct_actions[key], tool, params)
            if matched:
                outcome = self._apply_action(matched, "recovery")
                self.remediation_history.append((tool, target, params, "recovery"))
                return outcome

        # 2. Check partial actions
        if key in self._partial_actions:
            matched = self._match_action_in_list(self._partial_actions[key], tool, params)
            if matched:
                outcome = self._apply_action(matched, "partial")
                self.remediation_history.append((tool, target, params, "partial"))
                return outcome

        # 3. Check trap actions
        if key in self._trap_actions:
            matched = self._match_action_in_list(self._trap_actions[key], tool, params)
            if matched:
                outcome = self._apply_action(matched, "worsened")
                self.harm_events.append((tool, matched.get("message", "harmful action")))
                self.remediation_history.append((tool, target, params, "worsened"))
                return outcome

        # 4. For execute_runbook with unrecognized action: give helpful feedback
        if tool == "execute_runbook":
            agent_action = self._extract_action_name(params)
            svc_info = self.get_service_info(target)
            if svc_info and agent_action:
                available = svc_info.get("available_actions", [])
                if agent_action not in [a.lower() for a in available]:
                    self.remediation_history.append((tool, target, params, "no_effect"))
                    return RemediationOutcome(
                        outcome="no_effect",
                        message=f"Action '{agent_action}' is not available on {target}. Available actions: {', '.join(available[:5])}...",
                        post_state=self.system_state,
                    )

        # 5. Default — no effect
        default = self.remediation_data.get("default_response", {})
        self.remediation_history.append((tool, target, params, "no_effect"))
        return RemediationOutcome(
            outcome="no_effect",
            message=default.get("message", f"Action had no observable effect on the incident."),
            post_state=self.system_state,
        )

    def _params_match(self, required: Dict, provided: Dict) -> bool:
        """Check if provided params match required params.

        Empty required = any params accepted.
        For execute_runbook, checks action name + nested params.
        """
        if not required:
            return True
        for key, value in required.items():
            if value is None:
                continue  # None = any value accepted for this key
            provided_val = provided.get(key)
            if provided_val is None:
                return False
            if str(provided_val).lower() != str(value).lower():
                return False
        return True

    def _apply_action(self, action: Dict, outcome_type: str) -> RemediationOutcome:
        """Apply an action definition and update state."""
        post_state = action.get("post_state", self.system_state)
        self.system_state = post_state

        # Generate overlay logs
        action_time = self.base_time + timedelta(minutes=self.duration_minutes)
        for log_def in action.get("post_logs", []):
            offset = log_def.get("offset_after_action_seconds", 5)
            ts = action_time + timedelta(seconds=offset)

            # Render template with variables
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

        # Store overlay metrics
        for svc, metrics in action.get("post_metrics", {}).items():
            if svc not in self.overlay_metrics:
                self.overlay_metrics[svc] = {}
            self.overlay_metrics[svc].update(metrics)

        return RemediationOutcome(
            outcome=outcome_type,
            message=action.get("message", "Action completed."),
            post_state=post_state,
            post_logs=action.get("post_logs", []),
            post_metrics=action.get("post_metrics", {}),
        )

    def is_resolved(self) -> bool:
        """Check if the system has been fixed."""
        return self.system_state in ("healthy", "recovering")
