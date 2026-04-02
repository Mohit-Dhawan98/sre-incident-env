# Scenario Difficulty Calibration

## Why Graph Depth Doesn't Determine Difficulty

Our initial assumption was: deeper causal chains = harder scenarios. This is wrong.

A 6-hop chain where every log says "upstream service-X timeout" is trivial for LLMs — they just follow the breadcrumbs. A 1-hop scenario where the root cause is a subtle config drift with 5 services all showing errors simultaneously is much harder.

## The 4 Dimensions of Difficulty

Each scenario is scored 0-2 on four dimensions. Total score determines difficulty tier.

### Dimension 1: Root Visibility (0-2)
How loudly does the root cause service announce its failure?

| Score | Criteria | Example |
|-------|----------|---------|
| 0 (easy) | Root has 4+ ERROR/WARN logs | worker-pool with 10 OOM logs |
| 1 (medium) | Root has 2-3 ERROR/WARN logs | wal-archiver with 3 disk warnings |
| 2 (hard) | Root has 0-1 logs, or only INFO | ntp-client with 1 subtle WARN |

### Dimension 2: Traceability (0-2)
Do downstream services point back to the root by name?

| Score | Criteria | Example |
|-------|----------|---------|
| 0 (easy) | Other logs mention root by name + multiple upstream references | "upstream payment-processor: HTTP 503" |
| 1 (medium) | Root mentioned once OR generic upstream references | "upstream timeout" without naming who |
| 2 (hard) | No service names in error logs, must infer from metrics | Logs just say "HTTP 503" or "connection refused" |

### Dimension 3: Signal-to-Noise (0-2)
How many services are screaming, and how many are red herrings?

| Score | Criteria | Example |
|-------|----------|---------|
| 0 (easy) | ≤3 services show errors, 0 red herrings | Only root + 2 downstream show errors |
| 1 (medium) | ≤4 services show errors, ≤1 red herring | 4 services erroring, 1 is background job |
| 2 (hard) | 5+ services show errors, 2+ red herrings | Everything looks broken, multiple coincidental issues |

### Dimension 4: Failure Familiarity (0-2)
Is this a failure pattern that LLMs have seen in training data?

| Score | Criteria | Examples |
|-------|----------|---------|
| 0 (easy) | Well-known, dramatic pattern | OOM kill, connection pool exhausted, cert expired, disk full, N+1 query |
| 1 (medium) | Recognizable but less common | Cache stampede, connection leak, GC pressure, replication lag, rate limit |
| 2 (hard) | Obscure, subtle mechanism | Config drift (session timeout changed), clock skew → JWT rejection, library version conflict in serialization, bad index drop, DNS stub zone misconfiguration |

## Scoring

| Total Score | Difficulty | Budget |
|-------------|-----------|--------|
| 0-2 | Easy | 8 queries |
| 3-4 | Medium | 10 queries |
| 5-6 | Hard | 12 queries |
| 7-8 | Expert | 15 queries |

**Budget is inverted**: harder incidents get MORE queries, like a real P0 getting all hands on deck. Difficulty comes from content complexity, not resource starvation.

## Current Distribution

| Difficulty | Count | Score Range |
|------------|-------|-------------|
| Easy | 4 | 1-2 |
| Medium | 12 | 3-4 |
| Hard | 11 | 5-6 |
| Expert | 0 | 7-8 (none yet — would need invisible root + no traceability + high noise + obscure mechanism) |

## How to Create Expert Scenarios

An expert scenario (score 7-8) would need ALL four dimensions at max:
- Root service has NO error logs (invisible)
- No service names in any error messages (no tracing)
- 6+ services showing errors with 3+ red herrings (maximum noise)
- Obscure failure mechanism (e.g., kernel parameter drift, serialization format change, race condition in distributed lock)
