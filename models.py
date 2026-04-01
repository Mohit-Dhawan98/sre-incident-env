"""
Data models for SRE Incident Response Environment.

MCP pattern: no custom Action class needed — framework provides CallToolAction.
We define Observation and State for typed environment responses.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    timestamp: str
    service: str
    level: str
    message: str


class MetricPoint(BaseModel):
    timestamp: str
    value: float


class SREObservation(BaseModel):
    """Observation returned after each environment interaction."""

    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # SRE-specific fields
    action_type: str = ""
    logs: Optional[List[LogEntry]] = None
    metric_series: Optional[List[MetricPoint]] = None
    services: Optional[List[str]] = None
    message: str = ""
    queries_remaining: int = 0
    episode_elapsed_seconds: float = 0.0


class SREState(BaseModel):
    """Episode-level metadata exposed via GET /state."""

    episode_id: str = ""
    step_count: int = 0
    scenario_id: str = ""
    difficulty: str = ""
    services: List[str] = Field(default_factory=list)
    queries_used: int = 0
    query_budget: int = 0
    diagnosis_submitted: bool = False
    current_reward: float = 0.0
