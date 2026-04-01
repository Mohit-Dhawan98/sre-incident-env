"""
SRE Incident Response Environment Client.

Connects to the SRE Incident Response environment server via MCP.

Example:
    >>> with SREIncidentEnv(base_url="http://localhost:8000") as env:
    ...     env.reset(difficulty="medium")
    ...     tools = env.list_tools()
    ...     result = env.call_tool("list_services")
    ...     result = env.call_tool("read_logs", service="api-gateway", window_minutes=10)
    ...     result = env.call_tool("submit_diagnosis",
    ...         root_cause="connection pool exhausted",
    ...         affected_service="db-primary",
    ...         confidence=0.8)
"""

from openenv.core.mcp_client import MCPToolClient


class SREIncidentEnv(MCPToolClient):
    """Client for the SRE Incident Response Environment.

    Inherits all functionality from MCPToolClient:
    - list_tools(): Discover available investigation tools
    - call_tool(name, **kwargs): Call a tool by name
    - reset(**kwargs): Reset the environment (start new incident)
    - step(action): Execute an action (for advanced use)
    """

    pass
