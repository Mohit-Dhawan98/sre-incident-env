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
)


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
