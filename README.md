---
title: SRE Incident Response Environment
emoji: 🔧
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
---

# SRE Incident Response Environment

An OpenEnv-compliant reinforcement learning environment that simulates on-call Site Reliability Engineering. An AI agent receives a production incident alert, investigates by querying service logs and metrics across a multi-service architecture, reasons about causality through red herrings and cascading failures, and submits a root-cause diagnosis. The environment scores it automatically using embedding-based semantic similarity — zero LLM calls at runtime.

## Quick Start

```bash
# Install
pip install git+https://huggingface.co/spaces/<org>/sre-incident-env

# Or clone and install locally
git clone <repo-url> && cd sre-incident-env
uv sync

# Run the baseline agent
export OPENAI_API_KEY=your_key
python inference.py --difficulty medium --episodes 3
```

```python
from client import SREIncidentEnv

with SREIncidentEnv(base_url="http://localhost:8000") as env:
    env.reset(difficulty="medium")
    tools = env.list_tools()
    result = env.call_tool("list_services")
    result = env.call_tool("read_logs", service="api-gateway", window_minutes=10)
    result = env.call_tool("submit_diagnosis",
        root_cause="Redis node failure caused cache stampede",
        affected_service="product-cache",
        confidence=0.85)
```

## How an Agent Interacts

The environment exposes 4 MCP tools:

| Tool | Cost | Description |
|------|------|-------------|
| `list_services` | Free | Returns service names in the incident topology |
| `read_logs(service, window_minutes, level_filter)` | 1 query | Returns log entries for a service |
| `check_metric(service, metric, window_minutes)` | 1 query | Returns metric time-series (30s resolution) |
| `submit_diagnosis(root_cause, affected_service, confidence)` | Ends episode | Submits root-cause diagnosis, returns reward |

**Observation** after each tool call includes the tool result and `queries_remaining` countdown.

**Episode ends** when: the agent calls `submit_diagnosis`, OR the query budget is exhausted (reward = 0).

## Difficulty Tiers

| Tier | Query Budget | Red Herrings | Causal Hops | Metric-Log Tension | Special |
|------|-------------|--------------|-------------|-------------------|---------|
| easy | 15 | 0 | 2 | No | Dependency graph shown |
| medium | 10 | 1 | 2-3 | No | Deploy red herring |
| hard | 7 | 2 | 3-4 | Yes | Logs and metrics disagree |
| expert | 5 | 3 | 4+ | Yes | + phantom service in logs |

## Reward Function

5-component reward, all computed locally (zero LLM calls):

| Component | Weight | Method |
|-----------|--------|--------|
| Service accuracy | 0.25 | Exact match: did you identify the correct root service? |
| Semantic similarity | 0.45 | Embedding cosine similarity (EmbeddingGemma-300M) between submitted and true root cause |
| Investigation coverage | 0.15 | Fraction of relevant services investigated before diagnosing |
| Efficiency | 0.10 | Non-linear: full bonus if ≤70% budget used, penalizes only budget exhaustion |
| Confidence calibration | 0.05 | Penalizes confident wrong answers, rewards well-calibrated confidence |

Difficulty multiplier: easy=0.6, medium=0.8, hard=1.0, expert=1.2. Total clamped to [0.0, 1.0].

## Custom Incident Registry

Bring your own scenarios:

```bash
export OPENENV_CUSTOM_REGISTRY=/path/to/my_incidents.jsonl
```

Each line is a JSON object following the schema in [`scenarios/schema.md`](scenarios/schema.md). Key fields:

```json
{
  "id": "my_scenario_001",
  "title": "Alert title",
  "difficulty": "medium",
  "duration_minutes": 15,
  "services": {"svc-a": {"upstream": []}, "svc-b": {"upstream": ["svc-a"]}},
  "failure": {
    "root_service": "svc-b",
    "root_cause_type": "connection_pool_exhausted",
    "root_cause_statement": "svc-b connection pool exhausted due to..."
  },
  "log_templates": [...],
  "metric_templates": {...}
}
```

## Episode Flow

```
Agent                              Environment
  │                                     │
  │──── reset(difficulty="hard") ──────>│  Pick scenario, generate logs/metrics
  │<─── alert message, budget=7 ────────│
  │                                     │
  │──── list_services() ───────────────>│  (free)
  │<─── ["api-gw","cache","db",...] ────│
  │                                     │
  │──── read_logs("api-gw", ERROR) ────>│  budget: 7→6
  │<─── [{ts, svc, level, msg},...] ────│
  │                                     │
  │──── check_metric("db","latency") ──>│  budget: 6→5
  │<─── [{ts: .., value: 22}, ...] ─────│
  │                                     │
  │  ... more investigation ...         │
  │                                     │
  │──── submit_diagnosis(cause,svc) ───>│  Compute reward, end episode
  │<─── reward=0.72, done=True ─────────│
```

## Run Locally

```bash
# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or use openenv
openenv serve

# Health check
curl http://localhost:8000/health
```

## Deploy to HuggingFace Spaces

```bash
# Build Docker image
docker build -t sre-incident-env .

# Push via openenv
openenv push --repo-id <your-org>/sre-incident-env
```

The Dockerfile pre-downloads the EmbeddingGemma-300M model at build time for fast cold starts.

## Benchmark Results

Tested across frontier models (April 2026):

| Model | Easy | Medium | Hard | Expert | Avg |
|-------|------|--------|------|--------|-----|
| o4-mini | 0.24 | 0.57 | 0.12 | 0.58 | 0.38 |
| gpt-5.4 | 0.38 | 0.51 | 0.10 | 0.02 | 0.25 |
| gpt-4o | 0.20 | - | - | - | 0.20 |

Key insight: even the best frontier models struggle on hard/expert scenarios, demonstrating significant room for RL training improvement.
