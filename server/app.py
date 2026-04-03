"""FastAPI application for SRE Incident Response Environment."""

import uvicorn
from openenv.core.env_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from server.environment import SREIncidentEnvironment

app = create_app(
    SREIncidentEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="sre_incident_env",
    max_concurrent_envs=5,
)


def main() -> None:
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=30,
        ws_ping_timeout=120,
    )


if __name__ == "__main__":
    main()
