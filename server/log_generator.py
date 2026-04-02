"""Procedural log generation from scenario templates."""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from models import LogEntry


class LogGenerator:
    def __init__(self, scenario: Dict[str, Any]):
        self.scenario = scenario
        self.duration_minutes = scenario["duration_minutes"]
        # Random base time shifted by up to 24h
        self.base_time = datetime(2026, 3, 15, 8, 0, 0) + timedelta(
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        self.entries: List[LogEntry] = []
        self._render_all()

    def _render_all(self) -> None:
        """Pre-render all log entries from templates."""
        for tmpl in self.scenario["log_templates"]:
            offset_s = tmpl["offset_seconds"]
            ts = self.base_time + timedelta(seconds=offset_s)
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{random.randint(0, 999):03d}Z"

            # Resolve template variables
            log_vars = tmpl.get("log_vars") or {}
            values: Dict[str, Any] = {"ts": ts_str}
            for var_name, var_def in log_vars.items():
                if "min" in var_def and "max" in var_def:
                    # Numeric range
                    if isinstance(var_def["min"], float) or isinstance(var_def["max"], float):
                        values[var_name] = round(
                            random.uniform(var_def["min"], var_def["max"]), 2
                        )
                    else:
                        values[var_name] = random.randint(
                            int(var_def["min"]), int(var_def["max"])
                        )
                elif "choices" in var_def:
                    values[var_name] = random.choice(var_def["choices"])

            # Render template
            try:
                message = tmpl["template"].format(**values)
            except KeyError:
                # If template has vars we don't know, render with ts only
                message = tmpl["template"].format_map(
                    _DefaultDict(values)
                )

            self.entries.append(
                LogEntry(
                    timestamp=ts_str,
                    service=tmpl["service"],
                    level=tmpl["level"],
                    message=message,
                )
            )

        # Sort by timestamp
        self.entries.sort(key=lambda e: e.timestamp)

    def get_logs(
        self,
        service: str,
        window_minutes: int = 5,
        level_filter: Optional[str] = None,
    ) -> List[LogEntry]:
        """Return logs for a service within a time window."""
        # Calculate window bounds
        window_end = self.base_time + timedelta(
            minutes=self.duration_minutes
        )
        window_start = window_end - timedelta(minutes=window_minutes)
        start_str = window_start.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = window_end.strftime("%Y-%m-%dT%H:%M:%S")

        results = []
        for entry in self.entries:
            # Filter by service
            if entry.service.lower() != service.lower():
                continue
            # Filter by time window
            if entry.timestamp < start_str or entry.timestamp > end_str:
                continue
            # Filter by level
            if level_filter and entry.level != level_filter.upper():
                continue
            results.append(entry)

        return results

    def get_all_logs(
        self,
        service: str,
        level_filter: Optional[str] = None,
    ) -> List[LogEntry]:
        """Return all logs for a service (full episode timeline)."""
        results = []
        for entry in self.entries:
            if entry.service.lower() != service.lower():
                continue
            if level_filter and entry.level != level_filter.upper():
                continue
            results.append(entry)
        return results


class _DefaultDict(dict):
    """Dict that returns '{key}' for missing keys in str.format_map."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"
