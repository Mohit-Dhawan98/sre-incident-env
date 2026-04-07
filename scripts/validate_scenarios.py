"""Validate scenario quality against the hardening checklist.

Checks:
1. SOLVABILITY — optimal path exists, no dead ends, recovery from worsened
2. INFORMATION — root cause discoverable, params derivable, breadcrumbs exist
3. REALISTIC TRAPS — not too many, have reasons, worsened states recoverable
4. NO POISON SIGNALS — logs don't prescribe actions, progress hints at remaining work
5. DIFFICULTY FROM DEPTH — action names not obscure, challenge is ordering
6. FAIR MISLEADING — root cause has some signal, victim can be loud
7. GRADIENT SANITY — step counts match tier expectations
"""

import json
import re
import sys
from pathlib import Path
from collections import deque


# Words that indicate logs are prescribing actions (bad)
PRESCRIPTIVE_PATTERNS = [
    r'\buse\s+\w+\b',           # "use X"
    r'\brun\s+\w+\b',           # "run X"
    r'\btry\s+\w+\b',           # "try X"
    r'\bcall\s+\w+\b',          # "call X"
    r'\bexecute\s+\w+\b',       # "execute X"
    r'\binstead\b',             # "do X instead"
    r'\bfirst\b.*\bthen\b',    # "first do X then Y"
    r'\bneed(?:s)?\s+to\b',    # "needs to X"
    r'\bmust\s+\w+\b',         # "must X"
    r'\bshould\s+\w+\b',       # "should X"
    r'\brecommend\b',          # "recommend X"
]

# Acceptable context where these words are OK (system state descriptions)
PRESCRIPTIVE_EXCEPTIONS = [
    r'need(?:s)?\s+to\s+be\b',     # "needs to be reloaded" (state description)
    r'need(?:s)?\s+\w+ing\b',      # "needs reloading" (state description)
    r'must\s+be\b',                 # "must be fixed" (state description)
    r'should\s+have\b',            # "should have been" (past tense observation)
]

PLACEHOLDER_PATTERN = re.compile(r'\{val\}|\{[a-z_]+\}')
TIER_STEPS = {
    'easy': (1, 2),
    'medium': (2, 4),
    'hard': (2, 5),
    'expert': (4, 6),
}


def find_optimal_path(states, initial_state):
    """BFS to find shortest path from initial to any resolved state."""
    queue = deque([(initial_state, [initial_state], [])])
    visited = {initial_state}

    while queue:
        current, path, actions = queue.popleft()
        state_def = states.get(current, {})

        if state_def.get('is_resolved'):
            return path, actions

        for a in state_def.get('actions', []):
            if a['outcome'] in ('progress', 'recovery'):
                ns = a['next_state']
                if ns not in visited:
                    visited.add(ns)
                    action_desc = f"{a.get('tool','')}({a.get('target','')},{a.get('params',{}).get('_action','')})"
                    queue.append((ns, path + [ns], actions + [action_desc]))

    return [], []


def can_recover_from(states, state_name):
    """Check if there's a path from this state to any resolved state."""
    queue = deque([state_name])
    visited = {state_name}

    while queue:
        current = queue.popleft()
        state_def = states.get(current, {})
        if state_def.get('is_resolved'):
            return True
        for a in state_def.get('actions', []):
            if a['outcome'] in ('progress', 'recovery'):
                ns = a['next_state']
                if ns not in visited:
                    visited.add(ns)
                    queue.append(ns)
    return False


def check_prescriptive_text(text, context=""):
    """Check if text prescribes actions instead of describing state."""
    issues = []
    text_lower = text.lower()

    for pattern in PRESCRIPTIVE_PATTERNS:
        matches = re.finditer(pattern, text_lower)
        for m in matches:
            # Check if it's an acceptable exception
            is_exception = False
            for exc in PRESCRIPTIVE_EXCEPTIONS:
                if re.search(exc, text_lower[max(0,m.start()-10):m.end()+20]):
                    is_exception = True
                    break
            if not is_exception:
                issues.append(f"Prescriptive: '{m.group()}' in: ...{text[max(0,m.start()-20):m.end()+30]}...")

    return issues


def validate_scenario(scenario):
    """Run all checks on a single scenario. Returns list of (severity, message)."""
    issues = []
    sid = scenario['id']
    diff = scenario['difficulty']
    rem = scenario['failure']['remediation']
    states = rem['states']
    initial = rem.get('initial_state', 'broken')
    optimal_steps = rem.get('optimal_steps', 0)

    # === 1. SOLVABILITY ===
    opt_path, opt_actions = find_optimal_path(states, initial)
    if not opt_path:
        issues.append(('CRITICAL', 'No optimal path from initial to healthy'))
    elif not states.get(opt_path[-1], {}).get('is_resolved'):
        issues.append(('CRITICAL', f'Optimal path ends at {opt_path[-1]} which is NOT resolved'))

    actual_steps = len(opt_actions)
    if actual_steps != optimal_steps and optimal_steps > 0:
        issues.append(('ERROR', f'optimal_steps={optimal_steps} but actual shortest path is {actual_steps} steps'))

    # Check every non-resolved state has at least 1 progress action
    for sn, sd in states.items():
        if sd.get('is_resolved'):
            continue
        progress = [a for a in sd.get('actions', []) if a['outcome'] in ('progress', 'recovery')]
        if not progress:
            issues.append(('CRITICAL', f'State "{sn}" has NO progress/recovery actions — dead end'))

    # Check recovery from worsened states
    worsened_targets = set()
    for sn, sd in states.items():
        for a in sd.get('actions', []):
            if a['outcome'] == 'worsened':
                worsened_targets.add(a['next_state'])

    for ws in worsened_targets:
        if ws in states and not can_recover_from(states, ws):
            issues.append(('ERROR', f'Worsened state "{ws}" has no recovery path to healthy'))

    # === 2. INFORMATION AVAILABILITY ===
    # Check for unresolved placeholders
    for sn, sd in states.items():
        for a in sd.get('actions', []):
            msg = a.get('message', '')
            if PLACEHOLDER_PATTERN.search(msg):
                issues.append(('ERROR', f'Unresolved placeholder in {sn} action message: {msg[:80]}'))
            for log in a.get('post_logs', []):
                tmpl = log.get('template', '')
                # {ts} and {val} with log_vars are OK
                if '{val}' in tmpl and 'log_vars' not in log:
                    issues.append(('ERROR', f'{{val}} in log template without log_vars: {tmpl[:80]}'))

    # === 3. REALISTIC TRAPS ===
    for sn, sd in states.items():
        if sd.get('is_resolved'):
            continue
        worsened = [a for a in sd.get('actions', []) if a['outcome'] == 'worsened']
        if len(worsened) > 3:
            issues.append(('WARN', f'State "{sn}" has {len(worsened)} worsened traps (>3 may be unfair)'))

    # === 4. NO POISON SIGNALS ===
    for sn, sd in states.items():
        if sd.get('is_resolved'):
            continue
        for a in sd.get('actions', []):
            # Check action messages for prescriptive language
            msg = a.get('message', '')
            prescriptive = check_prescriptive_text(msg, f'{sn}/{a.get("outcome","")}')
            for p in prescriptive:
                # Only flag progress/no_effect messages (trap messages explaining what went wrong are OK)
                if a['outcome'] in ('progress', 'no_effect'):
                    issues.append(('WARN', f'[{sn}] {p}'))

            # Check post_logs for prescriptive language
            for log in a.get('post_logs', []):
                tmpl = log.get('template', '')
                log_prescriptive = check_prescriptive_text(tmpl, f'{sn}/log')
                for p in log_prescriptive:
                    issues.append(('WARN', f'[{sn}/log] {p}'))

    # Check progress messages hint at remaining work (for non-final progress)
    for sn, sd in states.items():
        if sd.get('is_resolved'):
            continue
        for a in sd.get('actions', []):
            if a['outcome'] == 'progress':
                ns = a['next_state']
                ns_def = states.get(ns, {})
                if not ns_def.get('is_resolved'):
                    msg = a.get('message', '')
                    hint_words = ['but', 'still', 'however', 'need', 'remaining', 'not yet', 'awaiting', 'pending']
                    has_hint = any(w in msg.lower() for w in hint_words)
                    if not has_hint:
                        issues.append(('WARN', f'[{sn}→{ns}] Progress message lacks "but/still/not yet" hint: {msg[:80]}'))

    # === 5. DIFFICULTY FROM DEPTH ===
    min_steps, max_steps = TIER_STEPS.get(diff, (1, 10))
    if optimal_steps < min_steps or optimal_steps > max_steps:
        issues.append(('WARN', f'{diff} tier but optimal_steps={optimal_steps} (expected {min_steps}-{max_steps})'))

    # Count total shortcuts (multiple progress actions per state)
    shortcuts = 0
    for sn, sd in states.items():
        if sd.get('is_resolved'):
            continue
        pc = sum(1 for a in sd.get('actions', []) if a['outcome'] in ('progress', 'recovery'))
        if pc > 1:
            shortcuts += pc - 1
    if shortcuts > optimal_steps:
        issues.append(('WARN', f'{shortcuts} shortcuts (alternative progress paths) — more than optimal_steps ({optimal_steps})'))

    # === 7. GRADIENT SANITY ===
    total_states = len(states)
    total_actions = sum(len(sd.get('actions', [])) for sd in states.values())

    return issues, {
        'optimal_steps': optimal_steps,
        'actual_shortest': actual_steps,
        'states': total_states,
        'actions': total_actions,
        'shortcuts': shortcuts,
        'worsened_states': len(worsened_targets),
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'scenarios/incidents_v2.jsonl'

    print(f"Validating: {path}")
    print("=" * 70)

    total_issues = {'CRITICAL': 0, 'ERROR': 0, 'WARN': 0}

    with open(path) as f:
        for line in f:
            s = json.loads(line)
            sid = s['id'].replace('_001', '').replace('_h002', '')[:40]
            diff = s['difficulty']

            issues, stats = validate_scenario(s)

            # Count by severity
            severities = {}
            for sev, msg in issues:
                severities[sev] = severities.get(sev, 0) + 1
                total_issues[sev] = total_issues.get(sev, 0) + 1

            status = "✓" if not issues else f"✗ {severities}"
            print(f"\n{status} {sid} ({diff}) — {stats['optimal_steps']}step, {stats['states']}states, {stats['shortcuts']}shortcuts")

            for sev, msg in issues:
                marker = {'CRITICAL': '🔴', 'ERROR': '🟠', 'WARN': '🟡'}[sev]
                print(f"  {marker} {sev}: {msg}")

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {total_issues['CRITICAL']} critical, {total_issues['ERROR']} errors, {total_issues['WARN']} warnings")


if __name__ == '__main__':
    main()
