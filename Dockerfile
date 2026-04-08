FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "pydantic>=2.0.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0" \
    "fastmcp>=3.0.0" \
    "openai>=2.7.2" \
    "requests>=2.31.0" \
    "python-dotenv" \
    "httpx>=0.27.0" \
    "gradio==6.10.0"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ENABLE_WEB_INTERFACE makes openenv-core auto-mount its own default Gradio
# inspector at /web and redirect / → /web, which shadows our custom
# landing UI. Our own Gradio mount in server/app.py provides the full UI.
ENV ENABLE_WEB_INTERFACE=false
CMD ["python", "-c", "import uvicorn; uvicorn.run('server.app:app', host='0.0.0.0', port=8000, ws_ping_interval=None, ws_ping_timeout=None)"]
