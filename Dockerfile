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
    "sentence-transformers>=2.7.0" \
    "pydantic>=2.0.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0" \
    "fastmcp>=3.0.0" \
    "openai>=2.7.2" \
    "requests>=2.31.0" \
    "python-dotenv"

# Pre-download embedding model at build time (needs HF_TOKEN for gated model)
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('google/embeddinggemma-300m')"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
