"""Data loaders for the Gradio landing UI.

Reads scenarios, leaderboard scores, and trace files at app startup
(no live inference — purely static display).
"""

from __future__ import annotations

import base64
import json
import re
import zlib
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_FILE = ROOT / "scenarios" / "incidents_v3.jsonl"
LEADERBOARD_LOG_DIR = ROOT / "outputs" / "hf_bench_v2"
TRACE_DIR = ROOT / "outputs" / "ui_traces"

MODELS = ["gpt-5.4", "o4-mini", "gpt-4o-mini"]

TIER_COLOR = {
    "easy":   "#06b6d4",  # cyan
    "medium": "#f59e0b",  # amber
    "hard":   "#a855f7",  # purple
}

OUTCOME_EMOJI = {
    "progress":  "✓",
    "recovery":  "★",
    "no_effect": "○",
    "worsened":  "⚠",
}

OUTCOME_COLOR = {
    "progress":  "#22c55e",  # green
    "recovery":  "#22c55e",
    "no_effect": "#94a3b8",  # gray
    "worsened":  "#ef4444",  # red
}


# ── Scenarios ──────────────────────────────────────────────────────


def load_scenarios() -> List[Dict[str, Any]]:
    """Load all scenarios from the v3 JSONL file."""
    out: List[Dict[str, Any]] = []
    with open(SCENARIO_FILE) as f:
        for line in f:
            s = json.loads(line)
            failure = s["failure"]
            rem = failure.get("remediation", {})
            out.append({
                "id": s["id"],
                "title": s["title"],
                "difficulty": s["difficulty"],
                "duration_minutes": s.get("duration_minutes", 15),
                "root_service": failure["root_service"],
                "root_cause_type": failure["root_cause_type"],
                "root_cause": failure["root_cause_statement"],
                "causal_chain": failure.get("causal_chain", []),
                "optimal_steps": rem.get("optimal_steps", 0),
                "services": list(s.get("services", {}).keys()),
                "states": rem.get("states", {}),
                "initial_state": rem.get("initial_state", "broken"),
                "resolved_states": rem.get("resolved_states", ["healthy"]),
            })
    return out


def scenarios_by_tier() -> Dict[str, List[Dict[str, Any]]]:
    """Group scenarios by difficulty tier."""
    groups: Dict[str, List[Dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for s in load_scenarios():
        groups.setdefault(s["difficulty"], []).append(s)
    return groups


# ── Mermaid state-graph generation ────────────────────────────────


def _action_label(action: Dict[str, Any]) -> str:
    """Short label for an action edge."""
    tool = action.get("tool", "")
    target = action.get("target", "")
    params = action.get("params", {}) or {}
    action_name = params.get("_action") or tool
    # Keep it compact
    return f"{action_name}<br/>({target})"


def _state_depths(states: Dict[str, Any], initial: str) -> Dict[str, int]:
    """BFS depth from initial state along progress/recovery edges."""
    depths = {initial: 0}
    q = deque([initial])
    while q:
        cur = q.popleft()
        sd = states.get(cur, {})
        for a in sd.get("actions", []):
            if a.get("outcome") not in ("progress", "recovery"):
                continue
            nxt = a.get("next_state")
            if nxt and nxt not in depths:
                depths[nxt] = depths[cur] + 1
                q.append(nxt)
    return depths


def mermaid_to_url(mermaid_code: str) -> str:
    """Encode mermaid source for the mermaid.ink SVG endpoint."""
    encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("ascii")
    encoded = encoded.rstrip("=")
    return f"https://mermaid.ink/svg/{encoded}?bgColor=0a0a0a"


# ── Inline SVG cache ──────────────────────────────────────────────
# Pre-fetch mermaid.ink SVGs at app startup and inline them in the
# rendered HTML. Avoids per-pageview network hits, CORS quirks, and
# flaky rendering inside the Gradio iframe.

_SVG_CACHE: Dict[str, str] = {}


def fetch_inline_svg(mermaid_code: str) -> str:
    """Fetch the mermaid.ink SVG for this source, cache it, return inline SVG.

    On failure returns a plain-text fallback inside a styled div so the
    card doesn't collapse.
    """
    cache_key = mermaid_code
    if cache_key in _SVG_CACHE:
        return _SVG_CACHE[cache_key]

    try:
        import httpx
        url = mermaid_to_url(mermaid_code)
        resp = httpx.get(url, timeout=15.0, follow_redirects=True)
        if resp.status_code == 200 and resp.text.lstrip().startswith("<svg"):
            svg = resp.text
            svg = re.sub(r'@import url\([^)]+\);', '', svg)
            extra = "max-width:100%;height:auto;display:block;margin:0 auto"
            if re.search(r'<svg[^>]*\sstyle="', svg):
                svg = re.sub(
                    r'(<svg[^>]*\sstyle=")([^"]*)(")',
                    lambda m: f'{m.group(1)}{extra};{m.group(2)}{m.group(3)}',
                    svg, count=1,
                )
            else:
                svg = re.sub(
                    r'<svg([^>]*)>',
                    rf'<svg\1 style="{extra}">',
                    svg, count=1,
                )
            _SVG_CACHE[cache_key] = svg
            return svg
    except Exception:
        pass

    fallback = (
        '<div style="color:#9ca3a3;font-family:monospace;font-size:11px;'
        'white-space:pre;text-align:left;padding:12px;">'
        + mermaid_code.replace("<", "&lt;").replace(">", "&gt;")
        + "</div>"
    )
    _SVG_CACHE[cache_key] = fallback
    return fallback


def warm_svg_cache() -> None:
    """Pre-fetch SVGs for all scenarios (called once at app startup)."""
    for s in load_scenarios():
        fetch_inline_svg(build_mermaid(s))


def build_mermaid(scenario: Dict[str, Any]) -> str:
    """Build a compact mermaid `graph LR` for the scenario state machine.

    - Progress/recovery path: green boxes + green arrows
    - Self-loop traps (wrong actions that keep you in same state): annotated
      as "[N traps]" on the source node label — since mermaid renders
      self-loops awkwardly, we show the count instead
    - Distinct-state traps (wrong actions that push to a named bad state like
      `etcd_crashed`): drawn as dashed red arrows to a red-bordered node
    - Orphan states (defined in data but unreachable from initial via any
      edge): hidden
    """
    states = scenario["states"]
    initial = scenario["initial_state"]
    resolved = set(scenario["resolved_states"])

    # BFS from initial via ANY outcome to find all genuinely reachable states
    reachable: Set[str] = {initial}
    queue = deque([initial])
    while queue:
        cur = queue.popleft()
        for a in states.get(cur, {}).get("actions", []):
            nxt = a.get("next_state")
            if nxt and nxt in states and nxt not in reachable:
                reachable.add(nxt)
                queue.append(nxt)

    # Progress depths (only progress/recovery edges) — used for node classification
    depths = _state_depths(states, initial)

    # Collect self-loop worsened actions per source state (with action names)
    self_loop_actions: Dict[str, List[str]] = defaultdict(list)
    # Collect distinct-state worsened edges (dedup by src,dst)
    distinct_edges: Set[Tuple[str, str]] = set()
    for name, sd in states.items():
        if name not in reachable:
            continue
        for a in sd.get("actions", []):
            if a.get("outcome") != "worsened":
                continue
            nxt = a.get("next_state")
            if not nxt or nxt not in reachable:
                continue
            if nxt == name:
                action_name = (a.get("params") or {}).get("_action") or a.get("tool", "?")
                self_loop_actions[name].append(action_name)
            else:
                distinct_edges.add((name, nxt))
    self_loop_counts = {k: len(v) for k, v in self_loop_actions.items()}

    # Build short IDs ONLY for reachable states
    id_map: Dict[str, str] = {}
    for i, name in enumerate(states.keys()):
        if name in reachable:
            id_map[name] = f"s{i}"

    lines = [
        "%%{init: {'theme':'dark','themeVariables':{'fontSize':'13px','fontFamily':'Inter, system-ui, sans-serif'},'flowchart':{'htmlLabels':true,'nodeSpacing':30,'rankSpacing':40}}}%%",
        "graph LR",
    ]

    # Node styling for real states (no ⚠N annotation — synthetic trap nodes
    # below carry that information)
    for name in states.keys():
        if name not in reachable:
            continue  # hide orphans
        nid = id_map[name]
        label = name.replace("_", " ")
        sd = states[name]
        if sd.get("is_resolved") or name in resolved:
            lines.append(f'{nid}(("✓ {label}")):::ok')
        elif name == initial:
            lines.append(f'{nid}["{label}"]:::bad')
        elif depths.get(name, 0) > 0:
            lines.append(f'{nid}["{label}"]:::prog')
        else:
            # Reachable only via worsened — a distinct trap state
            lines.append(f'{nid}["⚠ {label}"]:::trap')

    # ONE shared trap pool node per scenario, listing all distinct wrong-action
    # names. Each state with self-loop traps gets a single dashed arrow to it.
    # This consolidates the visual: agent sees "these actions are penalized"
    # in one place instead of repeated trap sinks.
    all_trap_actions: List[str] = []
    for actions in self_loop_actions.values():
        for act in actions:
            if act not in all_trap_actions:
                all_trap_actions.append(act)

    trap_pool_id: Optional[str] = None
    if all_trap_actions:
        trap_pool_id = "trap_pool"
        # Show up to 5 action names; rest as "+N more"
        shown = all_trap_actions[:5]
        suffix = ""
        if len(all_trap_actions) > 5:
            suffix = f"<br/>+{len(all_trap_actions)-5} more"
        label = "wrong actions<br/>(harm)<br/>" + "<br/>".join(shown) + suffix
        lines.append(f'{trap_pool_id}["{label}"]:::trap')

    # Progress/recovery edges + distinct-state worsened edges
    edge_idx = 0
    progress_indices: List[int] = []
    worsened_indices: List[int] = []
    seen_edges: Set[Tuple[str, str, str]] = set()
    for name, sd in states.items():
        if name not in reachable:
            continue
        for a in sd.get("actions", []):
            nxt = a.get("next_state")
            outcome = a.get("outcome", "no_effect")
            if not nxt or nxt not in reachable or nxt == name:
                continue
            if outcome == "no_effect":
                continue
            key = (name, nxt, outcome)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            if outcome == "worsened":
                lines.append(f"{id_map[name]} -.-> {id_map[nxt]}")
                worsened_indices.append(edge_idx)
            else:
                lines.append(f"{id_map[name]} ==> {id_map[nxt]}")
                progress_indices.append(edge_idx)
            edge_idx += 1

    # One dashed red arrow from each state with self-loop traps to the shared pool
    if trap_pool_id is not None:
        for src_name in self_loop_actions.keys():
            if src_name not in reachable:
                continue
            lines.append(f"{id_map[src_name]} -.-> {trap_pool_id}")
            worsened_indices.append(edge_idx)
            edge_idx += 1

    lines.extend([
        "classDef ok fill:#0d2818,stroke:#00d084,color:#00d084,stroke-width:3px",
        "classDef bad fill:#2a0e0e,stroke:#ff6b6b,color:#ffc4c4,stroke-width:3px",
        "classDef prog fill:#0f1f18,stroke:#00d084,color:#ededed,stroke-width:2px",
        "classDef trap fill:#2a0e0e,stroke:#ff6b6b,color:#ffc4c4,stroke-width:2px,stroke-dasharray:6 4",
    ])
    if worsened_indices:
        idx_list = ",".join(str(i) for i in worsened_indices)
        lines.append(f"linkStyle {idx_list} stroke:#ff6b6b,stroke-width:2px,stroke-dasharray:6 4")
    if progress_indices:
        idx_list = ",".join(str(i) for i in progress_indices)
        lines.append(f"linkStyle {idx_list} stroke:#00d084,stroke-width:2.5px")

    return "\n".join(lines)


# ── Leaderboard ────────────────────────────────────────────────────


END_RE = re.compile(r"\[END\] task=(\S+) score=([0-9.]+) steps=(\d+)")


def load_leaderboard() -> Dict[str, Dict[str, List[float]]]:
    """Parse leaderboard logs into {model: {scenario_id: [scores...]}}.

    Reads outputs/hf_bench_v2/<model>.log (latest HF runs).
    """
    out: Dict[str, Dict[str, List[float]]] = {m: defaultdict(list) for m in MODELS}
    for model in MODELS:
        log = LEADERBOARD_LOG_DIR / f"{model}.log"
        if not log.exists():
            continue
        with open(log) as f:
            for line in f:
                m = END_RE.match(line)
                if not m:
                    continue
                sid = m.group(1)
                score = float(m.group(2))
                out[model][sid].append(score)
    return {m: dict(s) for m, s in out.items()}


def leaderboard_averages() -> Dict[str, Dict[str, Any]]:
    """{model: {'per_scenario': {sid: avg}, 'per_tier': {tier: avg}, 'overall': avg}}"""
    raw = load_leaderboard()
    scenarios = {s["id"]: s["difficulty"] for s in load_scenarios()}
    out: Dict[str, Dict[str, Any]] = {}
    for model, scen_scores in raw.items():
        per_scen = {}
        per_tier: Dict[str, List[float]] = defaultdict(list)
        for sid, runs in scen_scores.items():
            if not runs:
                continue
            avg = sum(runs) / len(runs)
            per_scen[sid] = avg
            tier = scenarios.get(sid)
            if tier:
                per_tier[tier].append(avg)
        per_tier_avg = {t: sum(v) / len(v) for t, v in per_tier.items() if v}
        all_scores = [s for v in per_tier.values() for s in v]
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        out[model] = {
            "per_scenario": per_scen,
            "per_tier": per_tier_avg,
            "overall": overall,
        }
    return out


def score_color(score: float) -> str:
    """Color a cell in the leaderboard matrix by score."""
    if score >= 0.80:
        return "#16a34a"  # bright green
    if score >= 0.50:
        return "#22c55e"  # green
    if score >= 0.30:
        return "#f59e0b"  # amber
    if score >= 0.10:
        return "#f97316"  # orange
    return "#ef4444"       # red


# ── Traces ─────────────────────────────────────────────────────────


def load_trace(model: str, scenario_id: str, run: int = 1) -> Optional[Dict[str, Any]]:
    """Load a single trace file.

    Tries outputs/ui_traces/<model>/<scenario>_run<N>.json first,
    falls back to outputs/hardened_8_rerun/<scenario>_run<N>.json for gpt-5.4.
    """
    path = TRACE_DIR / model / f"{scenario_id}_run{run}.json"
    if not path.exists():
        fallback = ROOT / "outputs" / "hardened_8_rerun" / f"{scenario_id}_run{run}.json"
        if fallback.exists():
            path = fallback
        else:
            return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def summarize_trace(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract display-ready summary from a test_scenario.py trace JSON.

    The raw JSON has:
      {"summary": {scenario_id, reward, steps, elapsed_seconds, ...},
       "trace":   [msg, msg, msg ...]}
    where msg is a conversation message (role + content / tool_calls).
    """
    summary = trace_data.get("summary", {})
    steps: List[Dict[str, Any]] = []

    # Pair assistant tool_calls with the subsequent tool result
    trace = trace_data.get("trace", [])
    i = 0
    step_num = 0
    while i < len(trace):
        msg = trace[i]
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                tool_args = {}
            # Look for the next tool response
            outcome = None
            message = ""
            reward_at_step = None
            if i + 1 < len(trace) and trace[i + 1].get("role") == "tool":
                try:
                    parsed = json.loads(trace[i + 1]["content"])
                    outcome = parsed.get("outcome")
                    message = parsed.get("message", "") or parsed.get("error", "")
                    if "reward" in parsed:
                        reward_at_step = parsed["reward"]
                except (json.JSONDecodeError, TypeError):
                    pass
            step_num += 1
            steps.append({
                "n": step_num,
                "tool": tool_name,
                "args": tool_args,
                "outcome": outcome,
                "message": message[:280] if message else "",
                "reward": reward_at_step,
            })
            i += 2
        else:
            i += 1

    return {
        "scenario_id": summary.get("scenario_id", ""),
        "model": summary.get("model", ""),
        "reward": summary.get("reward"),
        "steps_used": summary.get("steps"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "done": summary.get("done", False),
        "error": summary.get("error"),
        "steps": steps,
    }


def available_traces() -> Dict[str, List[str]]:
    """Enumerate which (model, scenario) pairs have a trace file."""
    out: Dict[str, List[str]] = {m: [] for m in MODELS}
    scenario_ids = [s["id"] for s in load_scenarios()]
    for model in MODELS:
        for sid in scenario_ids:
            if load_trace(model, sid) is not None:
                out[model].append(sid)
    return out
