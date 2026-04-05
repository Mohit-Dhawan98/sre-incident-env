"""FastAPI application for SRE Incident Response Environment.

Exposes both:
- WebSocket /ws (openenv framework, works locally)
- HTTP /api/* (stateless per-request, works through HF Space proxy)

The HTTP API avoids long-lived WebSocket connections that get killed by
HF Space's reverse proxy (nginx/Cloudflare) idle timeout (~60s).
"""

import inspect
import json
import uuid
import asyncio
import uvicorn
from typing import Any, Dict, Optional

from fastapi import Body, HTTPException
from pydantic import BaseModel
from openenv.core.env_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.mcp_environment import get_server_tools

from server.environment import SREIncidentEnvironment

app = create_app(
    SREIncidentEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="sre_incident_env",
    max_concurrent_envs=5,
)


# ─── HTTP Session Management ─────────────────────────────────
# Each session holds a persistent SREIncidentEnvironment instance.
# No WebSocket needed — every call is a short HTTP round-trip.
# This avoids the HF Space proxy timeout that kills idle WebSockets.

_sessions: Dict[str, SREIncidentEnvironment] = {}
_session_lock = asyncio.Lock()


class ResetBody(BaseModel):
    difficulty: str = "medium"
    scenario_id: Optional[str] = None
    seed: Optional[int] = None


class CallToolBody(BaseModel):
    session_id: str
    tool_name: str
    arguments: Dict[str, Any] = {}


class CloseBody(BaseModel):
    session_id: str


@app.post("/api/reset")
async def api_reset(body: ResetBody = Body(default_factory=ResetBody)):
    """Create session, reset env, return session_id + tools + initial observation."""
    env = SREIncidentEnvironment()
    obs = env.reset(
        seed=body.seed,
        difficulty=body.difficulty,
        scenario_id=body.scenario_id,
    )
    session_id = str(uuid.uuid4())

    async with _session_lock:
        _sessions[session_id] = env

    # Get tool list from the MCP server
    tools = []
    server_tools = get_server_tools(env.mcp_server)
    for name, tool in server_tools.items():
        tools.append({
            "name": tool.name,
            "description": tool.description or "",
            "inputSchema": tool.parameters or {},
        })

    return {
        "session_id": session_id,
        "observation": obs.metadata,
        "done": obs.done,
        "reward": obs.reward,
        "tools": tools,
    }


@app.post("/api/call_tool")
async def api_call_tool(body: CallToolBody):
    """Call a tool on an existing session. Returns tool result + done + reward."""
    async with _session_lock:
        env = _sessions.get(body.session_id)

    if env is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {body.session_id}")

    # Call the tool function directly (same as /mcp endpoint does)
    server_tools = get_server_tools(env.mcp_server)
    if body.tool_name not in server_tools:
        raise HTTPException(
            status_code=400,
            detail=f"Tool not found: {body.tool_name}. Available: {list(server_tools.keys())}",
        )

    tool = server_tools[body.tool_name]
    try:
        if inspect.iscoroutinefunction(tool.fn):
            result = await tool.fn(**body.arguments)
        else:
            result = tool.fn(**body.arguments)
    except Exception as e:
        return {
            "result": json.dumps({"error": str(e)}),
            "done": env._done,
            "reward": env._current_reward,
        }

    # Track step count (normally done by step())
    env._steps += 1
    env._state.step_count = env._steps

    return {
        "result": result,
        "done": env._done,
        "reward": env._current_reward,
    }


@app.post("/api/close")
async def api_close(body: CloseBody):
    """Close and clean up a session."""
    async with _session_lock:
        env = _sessions.pop(body.session_id, None)

    if env is None:
        return {"closed": True, "session_id": body.session_id}

    env.close()
    return {"closed": True, "session_id": body.session_id}


def main() -> None:
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )


if __name__ == "__main__":
    main()
