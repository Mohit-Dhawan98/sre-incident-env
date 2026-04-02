"""Procedural metric time-series generation from scenario templates."""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from models import MetricPoint


class MetricGenerator:
    def __init__(self, scenario: Dict[str, Any], base_time: datetime):
        self.scenario = scenario
        self.duration_minutes = scenario["duration_minutes"]
        self.base_time = base_time
        self.resolution_seconds = 30
        # Pre-compute all series: {(service, metric_name): [MetricPoint]}
        self.series: Dict[Tuple[str, str], List[MetricPoint]] = {}
        self._render_all()

    def _render_all(self) -> None:
        for service, metrics in self.scenario["metric_templates"].items():
            for metric_name, config in metrics.items():
                points = self._generate_series(config)
                self.series[(service.lower(), metric_name.lower())] = points

    def _generate_series(self, config: Dict[str, Any]) -> List[MetricPoint]:
        total_points = (self.duration_minutes * 60) // self.resolution_seconds
        onset_point = int(
            config.get("onset_offset_minutes", 0) * 60 / self.resolution_seconds
        )
        noise_pct = config.get("noise_pct", 0.03)
        metric_type = config["type"]

        points = []
        for i in range(total_points + 1):
            ts = self.base_time + timedelta(seconds=i * self.resolution_seconds)
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

            if metric_type == "stable":
                value = config["value"]

            elif metric_type == "step":
                baseline = config["baseline"]
                after = config["after"]
                value = baseline if i < onset_point else after

            elif metric_type == "ramp":
                start = config["start"]
                end = config["end"]
                if i < onset_point:
                    value = start
                else:
                    progress = min(
                        1.0,
                        (i - onset_point) / max(1, total_points - onset_point),
                    )
                    value = start + (end - start) * progress

            elif metric_type == "spike":
                baseline = config["baseline"]
                peak = config["peak"]
                if i < onset_point:
                    value = baseline
                else:
                    # Sharp spike up, then partial recovery
                    elapsed = i - onset_point
                    spike_duration = max(1, (total_points - onset_point) // 3)
                    if elapsed < spike_duration:
                        # Rising
                        progress = elapsed / spike_duration
                        value = baseline + (peak - baseline) * progress
                    else:
                        # Partial recovery to ~40% of peak
                        recovery_progress = min(
                            1.0,
                            (elapsed - spike_duration)
                            / max(1, total_points - onset_point - spike_duration),
                        )
                        recovery_target = baseline + (peak - baseline) * 0.4
                        value = peak - (peak - recovery_target) * recovery_progress

            elif metric_type == "sawtooth":
                baseline = config.get("baseline", config.get("start", 0))
                peak = config.get("peak", config.get("end", 100))
                period = config.get("period_points", 10)
                if i < onset_point:
                    value = baseline
                else:
                    cycle_pos = (i - onset_point) % period
                    value = baseline + (peak - baseline) * (cycle_pos / period)

            else:
                value = config.get("value", config.get("baseline", 0))

            # Add gaussian noise
            if noise_pct > 0 and value != 0:
                noise = random.gauss(0, abs(value) * noise_pct)
                value += noise

            # Keep non-negative for most metrics
            value = max(0.0, value)
            points.append(MetricPoint(timestamp=ts_str, value=round(value, 4)))

        return points

    def get_metric(
        self,
        service: str,
        metric: str,
        window_minutes: int = 10,
    ) -> List[MetricPoint]:
        """Return metric series for a service within a time window."""
        # Case-insensitive lookup
        key = (service.lower(), metric.lower())
        if key not in self.series:
            # Try original case as fallback
            key = (service, metric)
            if key not in self.series:
                return []

        all_points = self.series[key]
        if not all_points:
            return []

        # Filter to last window_minutes of the episode
        window_end = self.base_time + timedelta(minutes=self.duration_minutes)
        window_start = window_end - timedelta(minutes=window_minutes)
        start_str = window_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = window_end.strftime("%Y-%m-%dT%H:%M:%SZ")

        return [p for p in all_points if start_str <= p.timestamp <= end_str]

    def list_metrics(self, service: str) -> List[str]:
        """Return available metric names for a service."""
        return [
            metric_name
            for (svc, metric_name) in self.series
            if svc == service.lower()
        ]
