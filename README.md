---
title: SRE Incident Response Environment
emoji: 🔧
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
---

# SRE Incident Response Environment

An OpenEnv RL environment that simulates on-call Site Reliability Engineering. An AI agent receives a production incident alert, investigates by querying service logs and metrics across a multi-service architecture, traces causal chains through red herrings and cascading failures, and submits a root-cause diagnosis. The environment scores it automatically — zero LLM calls at runtime, fully deterministic.

## Quick Start

```bash
pip install git+https://huggingface.co/spaces/Maverick98/sre-incident-env
```

```python
from client import SREIncidentEnv

async with SREIncidentEnv(base_url="https://Maverick98-sre-incident-env.hf.space") as env:
    await env.reset(difficulty="medium")
    tools = await env.list_tools()
    result = await env.call_tool("list_services")
    result = await env.call_tool("read_logs", service="api-gateway", window_minutes=10)
    result = await env.call_tool("submit_diagnosis",
        affected_service="product-cache",
        failure_type="cache_node_failure",
        root_cause="product-cache node failed causing checkout fallback",
        causal_chain="product-cache,checkout-service,api-gateway",
        confidence=0.85)
```

## How an Agent Interacts

The environment exposes 4 MCP tools:

| Tool | Description |
|------|-------------|
| `list_services` | Returns service names in the incident topology |
| `read_logs(service, window_minutes, level_filter)` | Returns log entries for a service |
| `check_metric(service, metric, window_minutes)` | Returns metric time-series (30s resolution) |
| `submit_diagnosis(affected_service, failure_type, root_cause, causal_chain, confidence)` | Submits diagnosis, returns reward, ends episode |

The agent has a generous query budget (100) — difficulty comes from scenario content, not resource constraints.

## Difficulty Tiers

| Tier | Count | Avg Score (frontier models) | What Makes It Hard |
|------|-------|---------------------------|-------------------|
| **Easy** | 5 | ~0.67 | Familiar failure patterns, clear signals, some investigation needed |
| **Medium** | 5 | ~0.39 | Multiple suspects, misleading red herrings, ambiguous metrics |
| **Hard** | 5 | ~0.32 | Invisible root, deep chains, obscure mechanisms, no breadcrumbs |
| **Expert** | 2 | ~0.24 | Invisible root + no metric clues + obscure infrastructure failures |

17 scenarios calibrated using joint consensus of o4-mini and gemini-2.5-flash. Hard/expert feature real production failures: NUMA cross-socket latency, CPU TSC drift, JVM metaspace exhaustion, Kafka partition rebalancing storms.

## Reward Function (V5)

7 components, all deterministic. No embedding models.

**Tier 1 — Did you solve it? (0.65):**

| Component | Weight | Method |
|-----------|--------|--------|
| Service identification | 0.25 | Exact match on root service (+ 0.08 adjacency partial) |
| Failure type | 0.15 | Exact match from 20-type taxonomy |
| Causal chain | 0.15 | F1 score vs golden propagation chain |
| Explanation keywords | 0.10 | Golden keyword checklist (5 terms per scenario) |

**Tier 2 — How did you solve it? (0.35):**

| Component | Weight | Method |
|-----------|--------|--------|
| Query efficiency | 0.20 | Actual vs per-scenario optimal queries |
| Investigation waste | 0.08 | Penalizes duplicate queries, tunnel vision |
| Investigation breadth | 0.07 | Fraction of causal-chain services investigated |

**Anti-gaming:** Components 2-4 are GATED on correct service identification. Wrong service = 0 on Tier 1. No free points.

## Baseline Scores

Reproducible scores from `inference.py` (1 episode per tier):

| Model | Easy | Medium | Hard | Expert | Overall |
|-------|------|--------|------|--------|---------|
| gpt-4o-mini | 0.91 | 0.42 | 0.28 | 0.15 | 0.44 |
| o4-mini | 0.84 | 0.48 | 0.34 | 0.26 | 0.48 |
| gemini-2.5-flash | 0.62 | 0.38 | 0.30 | 0.23 | 0.38 |

Hard/expert scenarios genuinely challenge frontier models. Easy scenarios are solvable with basic log-following.

## Action & Observation Spaces

**Action** (`CallToolAction`): Agent calls one of 4 MCP tools per step. Each tool has typed parameters.

**Observation** (`SREObservation`):
- `done: bool` — episode finished?
- `reward: float` — 0.0-1.0 (only non-zero on submit_diagnosis)
- `logs: List[LogEntry]` — `{timestamp, service, level, message}`
- `metric_series: List[MetricPoint]` — `{timestamp, value}`
- `services: List[str]` — service names (from list_services)
- `message: str` — alert text on reset, status messages

**State** (`SREState`): `episode_id`, `step_count`, `scenario_id`, `difficulty`, `services`, `queries_used`, `query_budget`, `diagnosis_submitted`, `current_reward`

## Episode Flow

```
Agent                              Environment
  │                                     │
  │──── reset(difficulty="hard") ──────>│  Pick scenario, generate logs/metrics
  │<─── alert message ─────────────────│
  │                                     │
  │──── list_services() ───────────────>│  (free)
  │<─── ["api-gw","cache","db",...] ────│
  │                                     │
  │──── read_logs("api-gw", ERROR) ────>│
  │<─── [{ts, svc, level, msg},...] ────│
  │                                     │
  │──── check_metric("db","latency") ──>│
  │<─── [{ts: .., value: 22}, ...] ─────│
  │                                     │
  │  ... investigate until confident ... │
  │                                     │
  │──── submit_diagnosis(cause,svc) ───>│  Compute reward, end episode
  │<─── reward=0.72, done=True ─────────│
```

## Run Locally

```bash
git clone https://huggingface.co/spaces/Maverick98/sre-incident-env
cd sre-incident-env
uv sync
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Run Inference

```bash
# Set required env vars
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"

# Run baseline (all 4 difficulties, 2 episodes each)
python inference.py

# Against HF Space
python inference.py --space https://Maverick98-sre-incident-env.hf.space

# Single difficulty
python inference.py --difficulty hard --episodes 3
```

## Architecture

- **Environment server**: MCPEnvironment (FastMCP) with 4 tools
- **Reward**: Fully deterministic, no model dependencies
- **Scenarios**: 17 (5 easy + 5 medium + 5 hard + 2 expert)
- **Client**: MCPToolClient (async/sync), installable via pip
- **Inference**: Native OpenAI function calling, smart context summarization
