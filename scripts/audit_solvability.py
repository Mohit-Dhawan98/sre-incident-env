"""Audit each scenario's solvability by tracing the optimal path
and checking at each step: does the agent have enough evidence to
figure out the next correct action?

For each state on the optimal path, checks:
1. Is the correct action discoverable in service_info?
2. If params are required, is the param VALUE findable in logs/metrics/service_info?
3. Does the progress message hint at what's still broken?
4. Are initial logs + post-action logs providing breadcrumbs?

Produces a per-scenario report showing the evidence trail.
"""

import json
import sys
from collections import deque
from pathlib import Path


def find_optimal_path(states, initial):
    """BFS shortest path to resolved state. Returns list of (state, action) tuples."""
    queue = deque([(initial, [])])
    visited = {initial}

    while queue:
        current, path = queue.popleft()
        state_def = states.get(current, {})

        if state_def.get('is_resolved'):
            return path

        for a in state_def.get('actions', []):
            if a['outcome'] in ('progress', 'recovery'):
                ns = a['next_state']
                if ns not in visited:
                    visited.add(ns)
                    queue.append((ns, path + [(current, a)]))

    return []


def audit_scenario(scenario):
    """Audit a single scenario for solvability."""
    sid = scenario['id']
    diff = scenario['difficulty']
    rem = scenario['failure']['remediation']
    states = rem['states']
    initial = rem.get('initial_state', 'broken')

    # Get optimal path
    opt_path = find_optimal_path(states, initial)
    if not opt_path:
        return f"CRITICAL: No path to healthy from {initial}"

    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"SCENARIO: {sid} ({diff})")
    lines.append(f"Optimal: {len(opt_path)} steps")
    lines.append(f"{'=' * 70}")

    # Collect ALL evidence available to agent
    # Initial logs (from log_templates)
    initial_logs = []
    for lt in scenario.get('log_templates', []):
        initial_logs.append({
            'service': lt['service'],
            'level': lt['level'],
            'template': lt['template'],
            'is_noise': lt.get('is_noise', False),
        })

    # Service info
    service_info = scenario.get('service_info', {})

    # Track cumulative post-action logs (what agent sees after each step)
    cumulative_post_logs = []

    for step_num, (state_name, action) in enumerate(opt_path, 1):
        tool = action.get('tool', '')
        target = action.get('target', '')
        params = action.get('params', {})
        action_name = params.get('_action', '') if params else ''
        other_params = {k: v for k, v in params.items() if k != '_action'} if params else {}

        lines.append(f"\n--- Step {step_num}: {state_name} → {action['next_state']} ---")
        lines.append(f"Action: {tool}({target}, {action_name})")
        if other_params:
            lines.append(f"Required params: {other_params}")

        # CHECK 1: Is the action discoverable?
        si = service_info.get(target, {})
        available_actions = si.get('available_actions', [])

        if tool == 'execute_runbook':
            if action_name in available_actions:
                lines.append(f"  [✓] '{action_name}' found in get_service_info({target}).available_actions")
            else:
                lines.append(f"  [✗] '{action_name}' NOT in get_service_info({target}).available_actions: {available_actions}")
        elif tool == 'restart_service':
            lines.append(f"  [✓] restart_service is always available")
        elif tool == 'rollback_deploy':
            lines.append(f"  [✓] rollback_deploy is always available")
        elif tool == 'scale_replicas':
            lines.append(f"  [✓] scale_replicas is always available")

        # CHECK 2: If params required, are values findable?
        if other_params:
            config_params = si.get('configurable_params', [])
            for key, value in other_params.items():
                # Check if key is in configurable_params
                key_found = key in config_params
                lines.append(f"  [{'✓' if key_found else '✗'}] Param key '{key}' in configurable_params: {key_found}")

                # Check if value is findable in logs/service_info
                value_str = str(value)
                # Search initial logs
                found_in = []
                for lt in initial_logs:
                    if value_str in lt['template']:
                        found_in.append(f"initial_log[{lt['service']}/{lt['level']}]")
                # Search post-action logs from previous steps
                for pl in cumulative_post_logs:
                    if value_str in pl.get('template', ''):
                        found_in.append(f"post_log[{pl.get('service','')}]")
                # Search service_info
                si_str = json.dumps(si)
                if value_str in si_str:
                    found_in.append(f"service_info[{target}]")
                # Search health checks
                for hc in si.get('health_checks', []):
                    if value_str in str(hc):
                        found_in.append(f"health_check[{target}]")

                if found_in:
                    lines.append(f"  [✓] Value '{value}' findable in: {', '.join(found_in)}")
                else:
                    # Check if value is DERIVABLE (e.g., 120 is 2x of 60 found in logs)
                    derivable = False
                    try:
                        v = int(value)
                        # Check if half, double, or nearby values exist
                        for lt in initial_logs + cumulative_post_logs:
                            tmpl = lt.get('template', '')
                            if str(v // 2) in tmpl or str(v * 2) in tmpl:
                                found_in.append(f"derivable(2x of value in logs)")
                                derivable = True
                                break
                    except (ValueError, TypeError):
                        pass

                    if derivable:
                        lines.append(f"  [~] Value '{value}' derivable: {', '.join(found_in)}")
                    else:
                        lines.append(f"  [✗] Value '{value}' NOT findable in any logs/service_info/metrics")

        # CHECK 3: What clues lead agent to this step?
        lines.append(f"  Evidence trail:")

        if step_num == 1:
            # First step — clues come from initial logs
            relevant_logs = [lt for lt in initial_logs if not lt['is_noise'] and target.lower() in lt['service'].lower()]
            if relevant_logs:
                for rl in relevant_logs[:3]:
                    lines.append(f"    initial_log [{rl['service']}/{rl['level']}]: {rl['template'][:100]}")
            else:
                # Check if any logs mention the target service
                any_mention = [lt for lt in initial_logs if not lt['is_noise'] and (
                    target.lower() in lt['template'].lower() or
                    action_name.lower().replace('_',' ') in lt['template'].lower()
                )]
                if any_mention:
                    for am in any_mention[:2]:
                        lines.append(f"    initial_log [{am['service']}/{am['level']}]: {am['template'][:100]}")
                else:
                    lines.append(f"    [!] No initial logs directly reference {target} or {action_name}")
        else:
            # Later steps — clues come from previous step's message/post_logs
            prev_state, prev_action = opt_path[step_num - 2]
            prev_msg = prev_action.get('message', '')
            lines.append(f"    prev_step msg: {prev_msg[:120]}")

            # Check if current action/target is hinted in prev message
            hints = []
            if target.lower() in prev_msg.lower():
                hints.append(f"target '{target}' mentioned")
            if action_name and action_name.replace('_', ' ').lower() in prev_msg.lower():
                hints.append(f"action '{action_name}' mentioned")
            # Check for state description words
            state_desc = states.get(state_name, {}).get('description', '')
            for word in ['still', 'but', 'pending', 'not yet', 'remains']:
                if word in prev_msg.lower():
                    hints.append(f"hint word '{word}' in prev message")
                    break

            if hints:
                lines.append(f"    [✓] Hints: {', '.join(hints)}")
            else:
                lines.append(f"    [!] No obvious hint toward this step in previous message")

        # Add this step's post_logs to cumulative
        for pl in action.get('post_logs', []):
            cumulative_post_logs.append(pl)

        # CHECK 4: Progress message hints at remaining work
        if action['outcome'] == 'progress':
            ns_def = states.get(action['next_state'], {})
            if not ns_def.get('is_resolved'):
                msg = action['message']
                hint_words = ['but', 'still', 'however', 'not yet', 'pending', 'remaining', 'awaiting']
                has_hint = any(w in msg.lower() for w in hint_words)
                lines.append(f"  [{'✓' if has_hint else '✗'}] Progress msg hints at remaining work: {has_hint}")

    # Final state
    final_state = opt_path[-1][1]['next_state']
    lines.append(f"\n→ Final state: {final_state} (resolved: {states.get(final_state, {}).get('is_resolved', False)})")

    return '\n'.join(lines)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'scenarios/incidents_v3.jsonl'

    # Filter to specific scenario if provided
    filter_id = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Auditing: {path}")
    if filter_id:
        print(f"Filter: {filter_id}")
    print()

    with open(path) as f:
        for line in f:
            s = json.loads(line)
            if filter_id and filter_id not in s['id']:
                continue
            report = audit_scenario(s)
            print(report)
            print()


if __name__ == '__main__':
    main()
