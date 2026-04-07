# V3 Scenario Hardening Rubric

## Process per scenario

### Step 1: Read the story
- Understand the incident as a real SRE would experience it
- Map the full state graph (every state, action, outcome)

### Step 2: Identify what's too easy
- Initial logs that hand the answer ("permission denied on /archive/old/")
- Progress messages that name the next target ("But cache-redis still has stale connections")
- Error messages that explain WHY instead of just WHAT ("Expected LZ4, found plain")
- show_* diagnostics that dump the full answer in one call
- Shortcuts (restart works as alternative to the correct action)

### Step 3: Apply hardness levers

**Lever 1: Opaque errors**
- Describe WHAT happened, not WHY
- ✗ "cleanup failed: format mismatch (expected lz4, found plain)"
- ✓ "cleanup: 342 processed, 12847 skipped"

**Lever 2: No target naming in progress messages**
- ✗ "But cache-redis still has stale connections"
- ✓ "But some downstream services still holding stale connections"

**Lever 3: Investigation required (split info across actions)**
- Don't dump the full answer in one diagnostic call
- Split across show_archive_status (volume stats) + validate_archive_integrity (format breakdown)
- Or bury in old logs (agent must widen time window)

**Lever 4: Time-windowed investigation**
- Root cause log placed at offset_seconds=-432000 (5 days ago)
- Agent must widen read_logs window_minutes to find it
- Breadcrumb: "config_last_modified: 5d ago" or "last_successful_cleanup: 5d ago"
- Models default to 5-60min windows — going to 7200 is a real SRE move

**Lever 5: Parameter precision**
- Correct values discoverable but require correlation
- Agent sees "format=lz4" in config + "plain_wal" in integrity scan → derives "set format to plain"
- Accept multiple reasonable values (plain, plain_wal) — don't reject correct reasoning

**Lever 6: Prevention step**
- System looks healthy after fix but root cause not prevented
- Agent must resist calling verify_resolution when things "look fixed"
- E.g., config_lock=enabled, disable_auto_tuning, fix_acme_dns_config

**Lever 7: Wrong-service traps**
- Victim services are plausible targets for restart/rollback
- restart(postgres-primary) when archive is full → crash recovery (worsened)
- restart(haproxy-lb) when TCP buffers are wrong → drops all connections (worsened)

### Step 4: Ensure solvability (NO GUESSWORK)
- Every correct action name in get_service_info available_actions
- Every required param KEY in configurable_params
- Every required param VALUE discoverable in:
  - Initial logs (possibly with wide time window)
  - Post-action logs from previous steps
  - show_* diagnostic actions
  - health_checks in service_info
  - OR derivable (120 = 2x of 60 found in logs)

### Step 5: Logs must read like real system output
- ✓ "[{ts}] cron.cleanup: exit 1. vol=97%"
- ✓ "archive_format=lz4 segments=13189 state=PAUSED"
- ✗ "Cleanup failed because permissions are broken. Fix permissions first."
- ✗ "Use issue_emergency_cert instead."

### Step 6: Validate with automated tools
```bash
# Checklist validator (prescriptive language, solvability, trap count)
uv run python scripts/validate_scenarios.py scenarios/incidents_v3.jsonl

# Solvability auditor (evidence trail at every step)
uv run python scripts/audit_solvability.py scenarios/incidents_v3.jsonl <scenario_filter>
```

### Step 7: Test with GPT-5.4
- Run 2 episodes per scenario
- Full args logged (no truncation)
- Target scores: easy 0.6-0.8, medium 0.4-0.6, hard 0.2-0.4, expert 0.1-0.2
- If model passes: apply more levers
- If model fails with 0.0: check if data issue (wrong param rejection) vs genuine difficulty

### Step 8: Check failure mode
- If 0.15 (didn't fix): is evidence trail complete? Did model see the clue?
- If model tried correct action with wrong params: accept the reasonable variant
- If model never found the root cause: is there a breadcrumb pointing to it?
- If completely stuck: add one more breadcrumb (factual, not prescriptive)

## Common data issues to avoid
- Rejecting reasonable param values (plain_wal when we only accept plain)
- Prescriptive messages in no_effect/progress ("Use X instead", "Need to Y")
- Poison logs implying fix already happened ("Recovering", "Resumed")
- Inconsistent health_checks (saying "exit code 2" when cleanup actually succeeds)
- show_archive_status changed 5 times across format

## Completed scenarios

| Scenario | Tier | v2→v3 | Levers used |
|----------|------|-------|-------------|
| cert_expiry | expert | 0.86→0.10 | Removed shortcuts, param precision, prevention step, no prescriptive |
| kernel_tcp | expert | 0.87→0.10 | Removed silver-platter log, no target naming, param precision, prevention |
| wal_archive | hard | 0.90→testing | Complete rewrite: format mismatch story, time-windowed investigation, split diagnostics, prevention |
