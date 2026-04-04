---
title: SRE Incident Response Environment
emoji: "\U0001F527"
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
---

# SRE Incident Response Environment

An OpenEnv RL environment that simulates the full on-call SRE lifecycle: investigate a production incident across a multi-service architecture, diagnose the root cause, apply multi-step remediation (where wrong actions make things worse), and verify resolution. The environment is a **state-graph maze** — the agent navigates through broken, degraded, and critical states to reach healthy.

17 real-world production incidents across 4 difficulty tiers. 96 states, 253 actions, trap doors at every intermediate state. Fully deterministic reward, zero LLM calls at runtime.

## Quick Start

```bash
pip install git+https://huggingface.co/spaces/Maverick98/sre-incident-env
```

```python
from client import SREIncidentEnv

async with SREIncidentEnv(base_url="https://Maverick98-sre-incident-env.hf.space") as env:
    obs = await env.reset(difficulty="medium")
    tools = await env.list_tools()

    # 1. Investigate
    result = await env.call_tool("list_services")
    result = await env.call_tool("read_logs", service="session-db", level_filter="ERROR")
    result = await env.call_tool("check_metric", service="session-db", metric="error_rate")

    # 2. Discover available actions
    result = await env.call_tool("get_service_info", service="session-db")

    # 3. Remediate (multi-step — order matters)
    result = await env.call_tool("execute_runbook", service="session-db",
        action="update_config", params='{"key": "connection_timeout", "value": "30000"}')
    result = await env.call_tool("read_logs", service="session-db")  # observe outcome

    result = await env.call_tool("execute_runbook", service="session-db",
        action="terminate_idle_connections")
    result = await env.call_tool("read_logs", service="session-db")  # observe outcome

    # 4. Verify resolution + submit diagnosis
    result = await env.call_tool("verify_resolution",
        affected_service="session-db",
        failure_type="config_drift",
        root_cause="session-db connection_timeout drifted to 300s causing idle connection buildup",
        causal_chain="session-db,user-service,api-gateway")
```

## Tools (10 MCP tools)

| Category | Tool | Description |
|----------|------|-------------|
| Discovery | `list_services` | Service topology (free, no query cost) |
| Investigation | `read_logs(service, window_minutes, level_filter)` | Logs for a service. Post-remediation logs reflect changed state |
| Investigation | `check_metric(service, metric, window_minutes)` | Metric time-series (30s resolution) |
| Discovery | `get_service_info(service)` | Service runbook: available actions, config params, recent deploys |
| Platform | `restart_service(service)` | Bounce service (clears runtime state, not config) |
| Platform | `rollback_deploy(service)` | Revert to previous deployment version |
| Platform | `scale_replicas(service, count)` | Scale horizontally |
| Application | `execute_runbook(service, action, params)` | Service-specific maintenance action (discovered via get_service_info) |
| Terminal | `verify_resolution(affected_service, failure_type, root_cause, causal_chain)` | Check system health + submit diagnosis. Ends episode |
| Terminal | `submit_diagnosis(...)` | V1 compat — diagnosis only, no remediation check |

## Episode Flow

```
Agent                                   Environment (State Machine)
  |                                          |
  |--- reset(difficulty="hard") ----------->|  Pick scenario, system_state = "broken"
  |<-- [INCIDENT ALERT] title + hint -------|
  |                                          |
  |--- list_services() ------------------->|  (free)
  |<-- ["api-gw","session-db","cache",...] --|
  |                                          |
  |--- read_logs("session-db", ERROR) ---->|  system_state: broken
  |<-- [{ts, svc, ERROR, "timeout 300s"}]---|
  |                                          |
  |--- get_service_info("session-db") ---->|  Discover: update_config, terminate_idle, ...
  |<-- {available_actions: [...], ...} ------|
  |                                          |
  |--- execute_runbook("session-db",       |
  |      "update_config", {timeout:30000}) >|  Correct step 1 -> system_state: "config_fixed"
  |<-- "pg_reload_conf() applied. ..." ------|
  |                                          |
  |--- read_logs("session-db") ----------->|  New logs show config applied
  |<-- [{INFO, "timeout now 30000ms"}] ------|
  |                                          |
  |--- execute_runbook("session-db",       |
  |      "terminate_idle_connections") ---->|  Correct step 2 -> system_state: "healthy"
  |<-- "47 idle connections terminated" ------|
  |                                          |
  |--- verify_resolution(diagnosis...) --->|  System healthy + grade diagnosis
  |<-- reward=0.92, done=True ---------------|
```

## State Graph Design

Each scenario defines a **directed graph of system states**. The agent navigates through it:

```
  [broken] --correct step 1--> [partially_fixed] --correct step 2--> [healthy]
     |                              |
     |--wrong action--> [critical]  |--wrong action--> [degraded]
     |                      |                              |
     |--no_effect--> [broken]       +---correct step 1---->+
```

- **Forward:** correct action in correct order advances state toward healthy
- **Sideways:** wrong action stays in same state (wasted step)
- **Backward:** trap action moves to critical/degraded (harder to recover)
- **Recovery:** even from critical, correct actions still lead forward (with penalty)

The agent discovers which direction it moved by **observing** (read_logs/check_metric after each action). The environment never tells the agent "you progressed" or "that was wrong" — it shows system state changes and the agent must infer.

### Difficulty = Steps + Investigation Depth + Trap Density

| Tier | Steps to Fix | Investigation | Traps per State | Total States |
|------|-------------|---------------|-----------------|-------------|
| Easy | 2 | Logs explicit | 1-2 | 3-4 |
| Medium | 2-3 | Logs hint at cause | 2-3 | 4-5 |
| Hard | 4 | Root cause hidden in noise | 3-4 | 5-6 |
| Expert | 5 | Root cause invisible in logs | 4-5 | 6-7 |

## Reward Function

### Design Philosophy

The reward is computed **at episode end** (terminal), not per-step. This is deliberate:

1. **Per-step rewards bias route selection.** An agent rewarded for "getting closer" learns to game proximity signals — moving toward the goal but getting stuck in dead ends. End-of-episode reward forces the agent to learn the complete path.
2. **The agent already gets per-step STATE feedback.** After every remediation, the agent can read_logs and check_metric to see what changed. This is the navigation signal — the reward is the final score.
3. **Per-step penalty is implicit.** More steps = lower efficiency score. The agent is incentivized to find the shortest correct path without an explicit -1/step.
4. **Partial credit exists.** An agent that investigates well but fails to fix still scores ~0.25. An agent that fixes but has bad diagnosis scores ~0.55. Full credit requires both.

### 7 Components (1.0 total)

| # | Component | Weight | What it measures |
|---|-----------|--------|-----------------|
| 1 | Investigation quality | 0.10 | Targeted investigation vs spray-and-pray |
| 2 | Reached exit | 0.35 | Did the system reach healthy state? |
| 3 | Path efficiency | 0.15 | Optimal steps vs actual remediation attempts |
| 4 | Trap avoidance | 0.15 | Avoided actions that worsened the system |
| 5 | Scouting behavior | 0.10 | Observed after fix (0.05) + discovered before action (0.05) |
| 6 | No wasted moves | 0.05 | Unique remediation actions vs total |
| 7 | Diagnosis quality | 0.10 | Service + type + keywords (GATED on system_healthy) |

### Anti-Gaming Properties

| Agent Strategy | Max Score | Why |
|----------------|-----------|-----|
| Skip investigation, guess fix | ~0.55 | Low investigation + may trigger traps |
| Investigate forever, never fix | ~0.25 | Investigation + trap avoidance, no exit/efficiency/diagnosis |
| Brute-force all tools | ~0.20 | Traps triggered + wasted moves + low efficiency |
| Fix correctly, bad diagnosis | ~0.55 | Exit + efficiency + traps, no diagnosis |
| Perfect: investigate, fix, diagnose | 1.00 | All components |

### Why Diagnosis is Gated on System Health

If you didn't fix the system, your diagnosis is untested. A correct diagnosis that wasn't acted on is worth 0 — this prevents agents from gaming diagnosis-only without attempting remediation.

## Scenarios (17)

| # | ID | Tier | Steps | Root Cause |
|---|-----|------|-------|-----------|
| 1 | circular_deadlock | Easy | 2 | Lock manager with deadlock detection disabled |
| 2 | connection_leak | Easy | 2 | API gateway middleware leaking file descriptors |
| 3 | connection_pool | Easy | 2 | Inventory service connection pool exhaustion |
| 4 | dns_misconfig | Easy | 2 | CoreDNS stub zone causing split-brain resolution |
| 5 | bad_index_drop | Easy | 2 | Dropped index on 2B-row analytics table |
| 6 | cache_stampede | Medium | 3 | Cold cache + thundering herd on product API |
| 7 | config_drift | Medium | 3 | Session DB timeout drifted from 30s to 300s |
| 8 | page_cache | Medium | 3 | Analytics engine sequential scan evicting page cache |
| 9 | thundering_herd | Medium | 3 | CDN cache purge + deploy causing origin flood |
| 10 | wal_disk_full | Medium | 2 | WAL archive permissions broken + disk full |
| 11 | cert_expiry | Hard | 4 | Mutual TLS cert expired on billing service |
| 12 | cpu_tsc_drift | Hard | 4 | CPU microcode update caused TSC clock drift |
| 13 | jvm_metaspace | Hard | 4 | Classloader agent leaking metaspace memory |
| 14 | kafka_rebalance | Hard | 4 | Zookeeper session timeout causing partition storms |
| 15 | numa_cross_socket | Hard | 4 | NUMA auto-migration causing cross-socket latency |
| 16 | etcd_compaction | Expert | 5 | etcd compaction backlog triggering quota alarm |
| 17 | kernel_tcp_rmem | Expert | 5 | Kernel TCP receive buffer silently dropping packets |

Each scenario has domain-specific service_info (Redis-specific actions for cache services, PostgreSQL-specific for databases, CoreDNS-specific for resolvers, etc.).

## Run Locally

```bash
git clone https://huggingface.co/spaces/Maverick98/sre-incident-env
cd sre-incident-env
uv sync
uv run server
```

## Run Inference

```bash
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-5.4"

# All 17 scenarios
python inference.py

# Against HF Space
python inference.py --space https://Maverick98-sre-incident-env.hf.space

# Single difficulty
python inference.py --difficulty hard --episodes 3
```

## Architecture

- **Environment server**: MCPEnvironment (FastMCP) with 10 MCP tools
- **State machine**: Graph-based traversal with per-state action tables
- **Reward**: Fully deterministic, 7 components, no model dependencies
- **Scenarios**: 17 built-in (5 easy + 5 medium + 5 hard + 2 expert), 96 states, 253 actions
- **Client**: MCPToolClient (async), WebSocket with configurable ping
- **Inference**: OpenAI function calling, smart context summarization
