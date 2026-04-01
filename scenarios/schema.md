# SRE Incident Environment — Scenario Schema & Design Guide

This document explains how to write scenarios for `incidents.jsonl`.
Each line is one complete JSON scenario object.

---

## Core Design Principles

### The 2-Hop Rule
The most visible symptom (the thing that triggers the alert) must be at least 2
causal hops from the actual root cause. The agent should NOT be able to find the
answer by grepping for the first ERROR line.

```
BAD:  db-primary crashes → API returns 500         (1 hop, trivial)
GOOD: deploy changes connection timeout config
      → payment-service holds connections open longer
      → db-primary connection pool exhausts
      → api-gateway upstream times out
      → users see checkout failures               (3 hops, interesting)
```

### Red Herrings Must Be Believable
A red herring is a service that shows degradation but is NOT the root cause.
The best red herrings are:
- A service that is a downstream VICTIM of the real failure (showing errors
  because its dependency is broken, not because it is broken)
- A service running a coincidentally timed background job that looks suspicious
- A recent deploy to a DIFFERENT service than the one that's actually broken

### Metric-Log Tension (hard + expert only)
At least one service should show a discrepancy between what its logs report and
what its metrics show. Examples:
- Logs show high error rate, but error_rate metric looks normal
  (log sampling was recently changed to sample at 10x rate — logs overstate reality)
- Metrics show CPU spike, but logs show no activity
  (a runaway goroutine with no logging)
- Latency metric looks fine, but logs show frequent timeouts
  (the metric is p50, the pain is in the p99 tail)

### Noise is Mandatory
Every scenario MUST include:
- Routine INFO logs from healthy services doing normal work
- At least one WARN that is a known-noisy alert teams ignore (disk usage at 72%,
  certificate expiring in 45 days, etc.)
- Background traffic in metrics even for services not involved in the failure

### The Phantom Service (expert only)
A service name appears in log messages (e.g. called by another service) but has
no entry in the `services` map. It exists in the incident story but not in the
observable graph. Agents waste queries trying to read logs for it.

---

## Field Reference

```
id                    string   Unique, snake_case. Include difficulty suffix: _001
title                 string   Incident title as it would appear in an alert
difficulty            string   "easy" | "medium" | "hard" | "expert"
duration_minutes      int      Timeline length (easy:10, medium:15, hard:20, expert:25)

services              object   Map of service_name → {upstream: [list of callers]}
                               upstream means "this service is called BY these services"
                               So api-gateway calls payment-service means:
                               payment-service.upstream = ["api-gateway"]

failure               object
  root_service        string   The service WHERE the bug lives
  root_cause_type     string   Short tag: oom_kill | connection_pool | config_drift |
                               slow_external_api | gc_pressure | disk_full |
                               n_plus_one_query | thread_pool_starvation | cert_expiry |
                               cache_stampede | replication_lag | rate_limit_breach
  onset_offset_minutes int     When the failure starts (minutes from episode start)
  root_cause_statement string  ONE sentence: [subject] [verb phrase] [caused what]
                               This is the ground truth the reward compares against.

red_herrings          array    List of {service, description, visible_in_logs: bool}
                               visible_in_logs means the agent CAN see their degradation
                               in log output (they just aren't the cause)

log_templates         array    Ordered list of log entries to render
  offset_seconds      int      Seconds from episode start when this log appears
  service             string   Must match a key in services map
  level               string   "INFO" | "WARN" | "ERROR" | "DEBUG"
  template            string   Python str.format() template. Variables:
                                 {ts}        — rendered timestamp
                                 {svc}       — randomized service name alias
                                 {val_*}     — numeric values (ranges in log_vars)
                                 {str_*}     — string values (choices in log_vars)
  log_vars            object   Optional. Variable ranges/choices:
                                 val_*: {min: N, max: N}
                                 str_*: {choices: ["a","b"]}
  is_noise            bool     true = this line is background noise, not incident-related

metric_templates      object   Map of service_name → metric_name → metric_config
                               metric_config fields:
                                 type: "stable" | "ramp" | "spike" | "step" | "sawtooth"
                                 baseline / start / end / peak values
                                 onset_offset_minutes: when shape begins
                                 noise_pct: gaussian noise as % of value (default 0.03)

tags                  array    Free-form tags for filtering/discovery
```

---

## Difficulty Matrix

| Property                  | easy | medium | hard | expert |
|---------------------------|------|--------|------|--------|
| Query budget              | 15   | 10     | 7    | 5      |
| Causal hops               | 2    | 2-3    | 3-4  | 4+     |
| Red herrings              | 0    | 1      | 2    | 3      |
| Log lines                 | ~40  | ~80    | ~120 | ~150   |
| Metric-log tension        | No   | No     | Yes  | Yes    |
| Service graph revealed    | Yes  | No     | No   | No     |
| Phantom service           | No   | No     | No   | Yes    |
| Deploy red herring        | No   | Yes    | Yes  | Yes    |
| Timeline length (minutes) | 10   | 15     | 20   | 25     |
