"""
SRE Incident Response Environment Client.

Two transport modes:
- SREIncidentEnv: WebSocket (original, works locally)
- SREIncidentEnvHTTP: HTTP (works through HF Space proxy, no idle timeout issues)

Both have identical API: reset(), list_tools(), call_tool().

Example (HTTP — recommended for HF Space):
    >>> async with SREIncidentEnvHTTP(base_url="https://Maverick98-sre-incident-env.hf.space") as env:
    ...     obs = await env.reset(difficulty="medium")
    ...     tools = await env.list_tools()
    ...     result = await env.call_tool("list_services")
    ...     result = await env.call_tool("read_logs", service="api-gateway")
    ...     result = await env.call_tool("verify_resolution",
    ...         affected_service="db-primary", failure_type="connection_leak",
    ...         root_cause="connection pool exhausted")

Example (WebSocket — works locally):
    >>> async with SREIncidentEnv(base_url="http://localhost:8000") as env:
    ...     obs = await env.reset(difficulty="medium")
    ...     tools = await env.list_tools()
    ...     result = await env.call_tool("list_services")
"""

import json
from typing import Any, Dict, List, Optional

from openenv.core.mcp_client import MCPToolClient


class SREIncidentEnv(MCPToolClient):
    """WebSocket-based client (original). Works locally, may timeout on HF Space."""

    async def _connect(self):
        """Override to set longer WebSocket ping timeout for slow LLM models."""
        from websockets.asyncio.client import connect as ws_connect

        if self._ws is not None:
            return

        self._ws = await ws_connect(
            self._ws_url,
            open_timeout=self._connect_timeout,
            max_size=getattr(self, "_max_message_size", 50 * 1024 * 1024),
            ping_interval=None,
            ping_timeout=None,
        )


class SREIncidentEnvHTTP:
    """HTTP-based client. Each call is a short HTTP round-trip.

    No WebSocket — avoids HF Space proxy idle timeout.
    Same API as SREIncidentEnv: reset(), list_tools(), call_tool().
    """

    def __init__(self, base_url: str, timeout: float = 120.0):
        # Normalize URL
        self.base_url = base_url.rstrip("/")
        if self.base_url.startswith("ws://"):
            self.base_url = self.base_url.replace("ws://", "http://", 1)
        elif self.base_url.startswith("wss://"):
            self.base_url = self.base_url.replace("wss://", "https://", 1)
        self.timeout = timeout
        self._client = None
        self._session_id: Optional[str] = None
        self._tools_cache: Optional[List[Dict]] = None

    async def __aenter__(self):
        import httpx
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args):
        if self._session_id and self._client:
            try:
                await self._client.post(
                    f"{self.base_url}/api/close",
                    json={"session_id": self._session_id},
                )
            except Exception:
                pass
        if self._client:
            await self._client.aclose()
            self._client = None

    async def reset(self, **kwargs) -> Dict[str, Any]:
        """Reset environment. Returns observation metadata dict."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=self.timeout)

        resp = await self._client.post(
            f"{self.base_url}/api/reset",
            json={
                "difficulty": kwargs.get("difficulty", "medium"),
                "scenario_id": kwargs.get("scenario_id"),
                "seed": kwargs.get("seed"),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._session_id = data["session_id"]
        self._tools_cache = data.get("tools", [])
        return data["observation"]

    async def list_tools(self) -> List[Dict]:
        """Return cached tool list from last reset()."""
        if self._tools_cache is not None:
            return self._tools_cache
        # If no cache, do a fresh reset to get tools
        return []

    async def call_tool(self, name: str, **kwargs) -> Any:
        """Call a tool. Returns the tool's result (usually a JSON string)."""
        if not self._session_id:
            raise RuntimeError("No active session. Call reset() first.")

        resp = await self._client.post(
            f"{self.base_url}/api/call_tool",
            json={
                "session_id": self._session_id,
                "tool_name": name,
                "arguments": kwargs,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        # Store done/reward on the response for callers that need it
        self._last_done = data.get("done", False)
        self._last_reward = data.get("reward", 0.0)

        return data.get("result", "")
