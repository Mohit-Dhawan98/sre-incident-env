---
title: SRE Incident Response Environment
emoji: "\U0001F527"
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
---

# SRE Incident Response Environment

An OpenEnv RL environment that simulates the full on-call SRE lifecycle: investigate a production incident across a multi-service architecture, diagnose the root cause, apply multi-step remediation (where wrong actions make things worse), and verify resolution.

The environment is a **state-graph maze** — the agent navigates through broken, degraded, and critical states to reach healthy. 17 real-world production incidents across 4 difficulty tiers. 96 states, 253 actions, trap doors at every intermediate state.

Fully deterministic reward, zero LLM calls at runtime.

## Baseline Scores

| Model | Easy | Medium | Hard | Expert | Overall | Fix Rate |
|-------|------|--------|------|--------|---------|----------|
| Gemini-2.5-Flash | 0.85 | 0.69 | 0.56 | 0.15 | 0.63 | 12/17 |
| GPT-5.4 | 0.74 | 0.70 | 0.58 | 0.15 | 0.61 | 11/17 |
| o4-mini | 0.74 | 0.55 | 0.51 | 0.15 | 0.55 | 10/17 |
| GPT-4o-mini | 0.29 | 0.29 | 0.13 | 0.15 | 0.22 | 2/17 |

Difficulty gradient verified across all 4 models: easy > medium > hard > expert. Expert scenarios (cert expiry, kernel TCP) unsolved by any model.

## Quick Start

```python
from client import SREIncidentEnvHTTP

async with SREIncidentEnvHTTP(base_url="https://Maverick98-sre-incident-env.hf.space") as env:
    obs = await env.reset(difficulty="medium")
    tools = await env.list_tools()

    # 1. Investigate
    result = await env.call_tool("list_services")
    result = await env.call_tool("read_logs", service="session-db", level_filter="ERROR")

    # 2. Discover available actions
    result = await env.call_tool("get_service_info", service="session-db")

    # 3. Remediate (multi-step, order matters)
    result = await env.call_tool("execute_runbook", service="session-db",
        action="update_config", params='{"key": "connection_timeout", "value": "30000"}')
    result = await env.call_tool("read_logs", service="session-db")  # observe outcome

    # 4. Verify + diagnose
    result = await env.call_tool("verify_resolution",
        affected_service="session-db", failure_type="config_drift",
        root_cause="session-db connection_timeout drifted to 300s",
        causal_chain="session-db,user-service,api-gateway")
```

## Tools (10 MCP tools)

| Category | Tool | Description |
|----------|------|-------------|
| Discovery | `list_services` | Service topology (free) |
| Investigation | `read_logs(service, window_minutes, level_filter)` | Logs. Post-remediation logs reflect changed state |
| Investigation | `check_metric(service, metric, window_minutes)` | Metric time-series |
| Discovery | `get_service_info(service)` | Service runbook: available actions, config params, recent deploys |
| Platform | `restart_service(service)` | Bounce service (clears runtime state, not config) |
| Platform | `rollback_deploy(service)` | Revert to previous deployment |
| Platform | `scale_replicas(service, count)` | Scale horizontally |
| Application | `execute_runbook(service, action, params)` | Service-specific action from runbook |
| Terminal | `verify_resolution(affected_service, failure_type, root_cause)` | Check health + submit diagnosis. Ends episode |

## State Graph Design

Each scenario defines a directed graph of system states. The agent navigates:

```
  [broken] --correct step--> [partially_fixed] --correct step--> [healthy]
     |                              |
     +--wrong action--> [critical]  +--wrong action--> [degraded]
```

- **Forward:** correct action advances toward healthy
- **Sideways:** wrong action wastes a step (no_effect)
- **Backward:** trap action worsens the system
- **Recovery:** even from critical, correct actions still lead forward (with penalty)

The agent discovers direction by **observing** (read_logs after each action). The environment never says "you progressed" — it shows system state changes.

## Action & Observation Spaces

**Action** (`CallToolAction`): Agent calls one of 10 MCP tools per step. Each tool has typed parameters (service name, metric name, action name, JSON params).

**Observation** (`Observation`):
- `done: bool` — episode finished?
- `reward: float` — 0.0 during episode, 0.0-1.0 on verify_resolution
- `metadata.logs: List[LogEntry]` — `{timestamp, service, level, message}`
- `metadata.metric_series: List[MetricPoint]` — `{timestamp, value}`
- `metadata.services: List[str]` — service names
- `metadata.message: str` — action outcome description, system state feedback
- `metadata.outcome: str` — remediation result (progress/recovery/no_effect/worsened)

**State**: `episode_id`, `step_count`, `scenario_id`, `difficulty`, `queries_used`, `query_budget`

## Reward Function

### 6 Components, 3 Dimensions, No Double Counting (perfect = 1.0)

Each step is evaluated on 3 independent dimensions: efficiency, safety, creativity.

| # | Component | Weight | Dimension | What it measures |
|---|-----------|--------|-----------|-----------------|
| 1 | Reached Exit | 0.35 | -- | Binary: did the system reach healthy? |
| 2 | Clean Path | 0.25 | Efficiency | Ratio: optimal_steps / actual_remediation |
| 3 | Diagnosis | 0.15 | -- | Service + type + keywords (gated on exit) |
| 4 | SRE Discipline | 0.10 | -- | Observe-after-fix + discover-before-action |
| 5 | Trap Avoidance | 0.10 | Safety | Full credit if no traps, -0.05 per worsened outcome |
| 6 | No Repeats | 0.05 | Creativity | Unique remediation actions / total |

### Design Principles

- **No double counting.** Each dimension is independent. Extra steps lower Clean Path ratio. Harmful steps lower Trap Avoidance. Repeated steps lower No Repeats. A step is never penalized twice on the same dimension.
- **Terminal reward, per-step state feedback.** Reward at episode end. Agent gets system state feedback (logs, metrics) after every action — that's the navigation signal.
- **Diagnosis gated on exit.** Untested diagnosis = 0. Prevents gaming diagnosis-only.
- **Difficulty gradient is natural.** Harder scenarios need more steps, more chances for waste, lower Clean Path ratio.

### Strategy Scores

| Strategy | Score |
|----------|-------|
| Perfect: investigate, fix optimally, diagnose | 1.00 |
| Fixed but bad diagnosis | ~0.85 |
| Fixed but 2x steps | ~0.62 |
| Good investigation, never fixed | ~0.15 |
| Brute force with traps | ~0.30 |

## Scenarios (17)

### Easy (2-step, 5 scenarios)
| Scenario | Root Cause |
|----------|-----------|
| circular_deadlock | Lock manager with deadlock detection disabled |
| connection_leak | API gateway middleware leaking file descriptors |
| connection_pool | Inventory service connection pool exhaustion |
| dns_misconfig | CoreDNS stub zone causing split-brain resolution |
| bad_index_drop | Dropped index on 2B-row analytics table |

### Medium (2-3 step, 5 scenarios)
| Scenario | Root Cause |
|----------|-----------|
| cache_stampede | Cold cache + thundering herd on product API |
| config_drift | Session DB timeout drifted from 30s to 300s |
| page_cache | Analytics engine sequential scan evicting page cache |
| thundering_herd | CDN cache purge + deploy causing origin flood |
| kafka_rebalance | Zookeeper session timeout causing partition storms |

### Hard (2-5 step, 5 scenarios)
| Scenario | Root Cause |
|----------|-----------|
| wal_disk_full | WAL archive permissions broken + disk full |
| cpu_tsc_drift | CPU microcode update caused TSC clock drift |
| jvm_metaspace | Classloader agent leaking metaspace memory |
| numa_cross_socket | NUMA auto-migration causing cross-socket latency |
| etcd_compaction | etcd compaction backlog triggering quota alarm |

### Expert (5 step, 2 scenarios)
| Scenario | Root Cause |
|----------|-----------|
| cert_expiry | Mutual TLS cert expired + ACME auto-renewal broken |
| kernel_tcp_rmem | Kernel TCP receive buffer silently dropping packets |

## Run Locally

```bash
git clone https://huggingface.co/spaces/Maverick98/sre-incident-env
cd sre-incident-env
uv sync
uv run server
```

## Run Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"

# Run baseline (2 episodes per tier, 8 total)
python inference.py --space https://Maverick98-sre-incident-env.hf.space

# Single difficulty
python inference.py --space https://Maverick98-sre-incident-env.hf.space --difficulty hard --episodes 3
```

## Architecture

- **Environment server**: MCPEnvironment (FastMCP) with 10 MCP tools
- **State machine**: Graph-based traversal with per-state action tables
- **Reward**: Fully deterministic, 6 components, no double counting
- **Scenarios**: 17 built-in (5 easy + 5 medium + 5 hard + 2 expert), 96 states, 253 actions
- **Client**: MCPToolClient (async), WebSocket with configurable ping
- **Inference**: OpenAI function calling, smart context summarization
