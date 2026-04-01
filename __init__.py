"""
SRE Incident Response Environment — OpenEnv RL Environment.

Simulates on-call SRE investigating production incidents.
Agent reads logs, checks metrics, and submits root-cause diagnosis.

Example:
    >>> from sre_incident_env import SREIncidentEnv
    >>>
    >>> with SREIncidentEnv(base_url="http://localhost:8000") as env:
    ...     env.reset(difficulty="medium")
    ...     tools = env.list_tools()
    ...     result = env.call_tool("list_services")
    ...     result = env.call_tool("read_logs", service="api-gateway")
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import SREIncidentEnv

__all__ = ["SREIncidentEnv", "CallToolAction", "ListToolsAction"]
