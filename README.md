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

The environment is a **state-graph maze** — the agent navigates through broken, degraded, and critical states to reach healthy. 8 real-world production incidents across 3 difficulty tiers. 50 states, 159 actions, trap doors at intermediate states.

Fully deterministic reward, zero LLM calls at runtime.

## Baseline Scores

Benchmarks against the deployed HF Space (8 scenarios × 2 runs = 16 episodes per model, except o4-mini at 1 run per scenario).

| Scenario | Tier | gpt-5.4 | o4-mini | gpt-4o-mini |
|---|---|---|---|---|
| `jvm_metaspace_classloader_leak_001` | easy | 0.59 | 0.89 | 0.17 |
| `etcd_compaction_quota_alarm_001` | easy | 0.60 | 0.00 | 0.05 |
| `kafka_partition_rebalance_storm_001` | medium | 0.34 | 0.27 | 0.00 |
| `cpu_microcode_tsc_drift_001` | medium | 0.30 | 0.23 | 0.16 |
| `cert_expiry_mutual_tls_001` | medium | 0.33 | 0.84 | 0.33 |
| `numa_cross_socket_latency_001` | hard | 0.89 | 0.13 | 0.14 |
| `kernel_tcp_rmem_silent_drop_001` | hard | 0.51 | 0.17 | 0.01 |
| `wal_archive_disk_full_h002` | hard | 0.04 | 0.00 | 0.00 |

### Tier averages

| Tier | gpt-5.4 | o4-mini | gpt-4o-mini |
|---|---|---|---|
| **easy** | 0.595 | 0.447 | 0.110 |
| **medium** | 0.322 | 0.447 | 0.165 |
| **hard** | 0.478 | 0.100 | 0.049 |
| **OVERALL** | **0.449** | **0.317** | **0.108** |

**Observations:**
- gpt-5.4 is the strongest overall. Solves `numa_cross_socket_latency` consistently (0.89) and picks up partial progress on every scenario.
- o4-mini (reasoning) has high run-to-run variance — when it commits to the right hypothesis it solves (jvm 0.89, cert_expiry 0.84), when it gets stuck in re-exploration it floors (etcd 0.00, kernel_tcp 0.17).
- gpt-4o-mini is the weak floor: consistently <0.20 on every scenario. Useful as a sanity-check baseline.
- `wal_archive_disk_full_h002` is the hardest scenario — unsolved by all three models. `kernel_tcp_rmem_silent_drop_001` is a close second.
- Gemini 2.5 Pro/Flash were tested but fail to converge on the multi-step tool-call protocol (Pro hit 503 rate limits; Flash looped on a single `rollback_deploy` call for 170+ steps). Excluded from baseline.

## Quick Start

```python
from client import SREIncidentEnvHTTP

async with SREIncidentEnvHTTP(base_url="https://Maverick98-sre-incident-env.hf.space") as env:
    obs = await env.reset(difficulty="medium")
    tools = await env.list_tools()

    # 1. Investigate
    result = await env.call_tool("list_services")
    result = await env.call_tool("read_logs", service="zookeeper", level_filter="WARN")

    # 2. Discover available actions
    result = await env.call_tool("get_service_info", service="zookeeper")

    # 3. Remediate (multi-step, order matters)
    result = await env.call_tool("execute_runbook", service="zookeeper",
        action="update_config", params='{"key": "session_timeout_ms", "value": "30000"}')
    result = await env.call_tool("read_logs", service="zookeeper")  # observe outcome

    # 4. Verify + diagnose
    result = await env.call_tool("verify_resolution",
        affected_service="zookeeper", failure_type="config_drift",
        root_cause="zookeeper session_timeout_ms reduced from 30000 to 6000 by automated config push, causing partition-manager session expirations")
```

## Tools (9 MCP tools)

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

- **Forward:** correct action advances toward healthy (`progress` or `recovery`)
- **Sideways:** wrong action wastes a step (`no_effect`)
- **Backward:** trap action worsens the system (`worsened`)
- **Recovery:** even from critical, correct actions still lead forward (with penalty)

The agent discovers direction by **observing** (read_logs, check_metric) after each action. The environment never says "you progressed" — it shows system state changes.

## Action & Observation Spaces

**Action** (`CallToolAction`): Agent calls one of 9 MCP tools per step. Each tool has typed parameters (service name, metric name, action name, JSON params).

**Observation** (`Observation`):
- `done: bool` — episode finished?
- `reward: float` — 0.0 during episode, final (0.0001, 0.9999) on `verify_resolution`
- `metadata.logs: List[LogEntry]` — `{timestamp, service, level, message}`
- `metadata.metric_series: List[MetricPoint]` — `{timestamp, value}`
- `metadata.services: List[str]` — service names
- `metadata.message: str` — action outcome description, system state feedback
- `metadata.outcome: str` — remediation result (`progress` / `recovery` / `no_effect` / `worsened`)

**State**: `episode_id`, `step_count`, `scenario_id`, `difficulty`, `queries_used`, `query_budget`

## Reward Function (V3)

### 7 components, quadratic partial progress (perfect = 1.00)

| # | Component | Weight | Gated on | What it measures |
|---|-----------|--------|----------|------------------|
| 1 | Exit | 0.40 | `system_healthy` | Binary: did the system reach healthy? |
| 2 | Progress | 0.20 | — | Quadratic: `(progress_depth / optimal_steps)²` — rewards partial-path credit |
| 3 | Diagnosis | 0.10 | `system_healthy` | Service (0.05) + type (0.025) + keywords (0.025) |
| 4 | Efficiency | 0.10 | progress | Ratio: `optimal_steps / actual_remediation`, scaled by progress |
| 5 | Discipline | 0.10 | progress | Observe-after-fix + discover-before-action, scaled by progress |
| 6 | Trap Avoidance | 0.05 | — | Starts full, −0.025 per worsened outcome |
| 7 | Diversity | 0.05 | progress | Unique remediation actions / total |

**Healthy max**: 0.40 + 0.20 + 0.10 + 0.10 + 0.10 + 0.05 + 0.05 = **1.00**
**Not-healthy max**: 0 + 0.20 + 0 + 0.10 + 0.10 + 0.05 + 0.05 = **0.50**

Final reward clamped strictly to `(0.0001, 0.9999)` for Hackathon Phase 2 validator compatibility.

### Design principles

- **Quadratic partial progress** — rewards agents that navigate part of the maze even if they don't reach healthy. Linear was too generous to half-solved runs; quadratic keeps the gap between fully-solved and partially-solved wide.
- **Gated diagnosis** — you only get diagnosis credit if the system is actually healthy. Prevents "I correctly described the bug but never fixed it" exploits.
- **No double counting** — each component measures a different dimension. Extra steps lower Efficiency. Harmful steps lower Trap Avoidance. Repeated steps lower Diversity.
- **Terminal reward, per-step observation feedback** — reward is computed at episode end via `verify_resolution`. During the episode the agent reads logs/metrics/system state as navigation signal.

## Scenarios (8)

### Easy (2 scenarios)
| ID | Root Cause |
|---|---|
| `jvm_metaspace_classloader_leak_001` | `classloader-agent` v1.12.0 enabled per-update compilation, leaking classloaders into metaspace on `data-enrichment`. Requires rollback → unload stale classloaders → circuit-breaker reset → dropped-event replay. |
| `etcd_compaction_quota_alarm_001` | `backup-agent` config management changed snapshot retention from 24h to 720h, filling etcd's 8GB quota and arming ALARM:NOSPACE. Requires retention fix → cleanup old snapshots → compaction → disarm alarm → resume scheduler. |

### Medium (3 scenarios)
| ID | Root Cause |
|---|---|
| `kafka_partition_rebalance_storm_001` | `zookeeper` `session_timeout_ms` auto-pushed from 30000 to 6000, causing partition-manager session expirations and rebalance storms. Requires config fix → force stable assignment → resume consumption → `config_lock` to prevent re-push. |
| `cpu_microcode_tsc_drift_001` | `tsc-firmware` microcode v2.8 introduced TSC clock drift on consensus-service hosts, destabilising Raft leader elections. Requires rollback → restart consensus → force leader election → retry pending state-store writes. |
| `cert_expiry_mutual_tls_001` | `cert-manager` ACME auto-renewal failed due to DNS-01 propagation timeout misconfig. billing-service mTLS cert expired. Requires emergency cert issue → TLS reload → session cache clear → fix ACME DNS config. |

### Hard (3 scenarios)
| ID | Root Cause |
|---|---|
| `numa_cross_socket_latency_001` | `numa-balancer` kernel auto-migration moved `matching-engine` cross-socket while memory stayed on the original node. Requires disable migration → pin process → compact order book → reconnect market data feed. |
| `kernel_tcp_rmem_silent_drop_001` | `sysctl-agent` aggressive tuning reduced `net.core.rmem_max`, causing silent TCP connection resets on haproxy-lb/nginx-ingress. Requires rollback tuning → reload network stack → disable auto-tuning. |
| `wal_archive_disk_full_h002` | `wal-archiver` archive-format mismatch (lz4 vs plain) caused archive failures → pg_wal growth → data volume exhaustion on `postgres-primary`. Multi-step recovery with trap actions. |

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
export MODEL_NAME="gpt-5.4"
export HF_TOKEN="your-api-key"

# Default: 1 run per scenario × 8 scenarios = 8 episodes (~8 min, under 20min validator cap)
python inference.py --space https://Maverick98-sre-incident-env.hf.space

# 2 runs per scenario (for variance estimates)
python inference.py --space https://Maverick98-sre-incident-env.hf.space --episodes 2

# Single tier
python inference.py --space https://Maverick98-sre-incident-env.hf.space --difficulty hard
```

Inference output follows the Hackathon Phase 2 structured format:

```
[START] task=jvm_metaspace_classloader_leak_001
[STEP] step=1 reward=0.0001
[STEP] step=2 reward=0.0001
...
[END] task=jvm_metaspace_classloader_leak_001 score=0.8917 steps=32
```

## Architecture

- **Environment server**: MCPEnvironment (FastMCP) with 9 MCP tools, HTTP transport
- **State machine**: Graph-based traversal with per-state action tables, max-progress-depth tracking for partial credit
- **Reward**: V3 7-component, fully deterministic, quadratic partial progress
- **Scenarios**: 8 hardened production incidents (2 easy + 3 medium + 3 hard), 50 states, 159 actions
- **Client**: `SREIncidentEnvHTTP` (async, HTTP transport — robust through HF Space proxy)
- **Inference**: OpenAI function calling via `openai` client, smart context summarization, per-episode session isolation
