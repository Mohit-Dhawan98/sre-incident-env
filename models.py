"""
Data models for SRE Incident Response Environment.

Follows the OpenEnv convention: custom Action/Observation types inheriting
from openenv base classes. Our types extend the MCP CallToolAction/Observation
with the same interface (thin wrappers for spec compliance).
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation


class SREAction(CallToolAction):
    """SRE incident response action — wraps MCP tool call."""
    pass


class SREObservation(CallToolObservation):
    """SRE incident response observation — wraps MCP tool result."""
    pass


class LogEntry(BaseModel):
    timestamp: str
    service: str
    level: str
    message: str


class MetricPoint(BaseModel):
    timestamp: str
    value: float


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
