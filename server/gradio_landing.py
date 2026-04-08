"""Gradio landing UI for the SRE Incident Response Environment.

Mounted at `/` via gr.mount_gradio_app(). Replaces the default openenv
inspector with a custom landing page featuring:

  1. Overview     — hero + uniqueness cards
  2. Scenarios    — 8 scenario cards with state-graph diagrams (mermaid.ink)
  3. Leaderboard  — model × scenario score matrix
  4. Traces       — interactive trace viewer for 3 frontier models
  5. Playground   — live interactive episode against the real environment
  6. Try It       — code samples

Design: GitHub Dark inspired — flat dark surface, single sky-blue accent,
system sans-serif typography.
"""

from __future__ import annotations

import inspect as _inspect
import json
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from server.environment import SREIncidentEnvironment
from server.ui_data import (
    MODELS,
    OUTCOME_EMOJI,
    available_traces,
    leaderboard_averages,
    load_scenarios,
    load_trace,
    score_color,
    summarize_trace,
)

try:
    from openenv.core.env_server.mcp_environment import get_server_tools
except ImportError:  # pragma: no cover
    get_server_tools = None  # type: ignore


# ── Theme CSS — GitHub Dark inspired ──────────────────────────────

CUSTOM_CSS = """
/* ─────────────────────────────────────────────────────────────
   Palette — Daytona emerald
   bg          #0a0a0a   near-black
   surface     #141414   card
   surface-2   #1a1a1a   nested card
   border      #262626   subtle
   text        #ededed   primary
   muted       #9ca3a3   secondary
   accent      #00d084   daytona emerald
   accent-dim  #00a866   darker emerald
   success     #00d084
   warning     #ffa94d
   danger      #ff6b6b
───────────────────────────────────────────────────────────── */

html, body, .gradio-container {
    background: #0a0a0a !important;
    color: #ededed !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "Helvetica Neue", Arial, sans-serif !important;
    font-feature-settings: "cv11", "ss01", "ss03";
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 28px 20px !important;
}
/* Tighten Gradio's default vertical gaps everywhere */
.gradio-container .gap { gap: 6px !important; }
.gradio-container .form, .gradio-container > .main { gap: 6px !important; padding: 0 !important; }
.gradio-container .block { margin: 0 !important; padding: 0 !important; }
/* Kill Gradio's outer wrapper top padding/margin */
.gradio-container > * { margin-top: 0 !important; }
.gradio-container > .main { padding-top: 0 !important; margin-top: 0 !important; }
body, html { margin: 0 !important; padding: 0 !important; }

/* ── Typographic scale (5 sizes) ─────────────────────────────
   xs    11px  meta labels, captions
   sm    13px  body, table cells, dropdowns
   base  14px  default
   lg    16px  card titles
   xl    20px  section h2, stat values
   2xl   26px  hero h1
*/
.gradio-container, .gradio-container p, .gradio-container span,
.gradio-container div, .gradio-container label, .gradio-container li,
.gradio-container td, .gradio-container th, .gradio-container strong,
.gradio-container summary {
    color: #ededed;
    font-size: 14px;
    line-height: 1.55;
}
.gradio-container h1, .gradio-container h2,
.gradio-container h3, .gradio-container h4 {
    color: #f5f5f5;
    font-weight: 600;
    letter-spacing: -0.015em;
    margin: 0;
    font-family: "Inter", -apple-system, sans-serif;
}
.gradio-container h1 { font-size: 26px; line-height: 1.2; font-weight: 700; }
.gradio-container h2 { font-size: 20px; line-height: 1.3; }
.gradio-container h3 { font-size: 15px; line-height: 1.35; }
.gradio-container h4 { font-size: 13px; line-height: 1.35; }
.gradio-container a { color: #00d084; text-decoration: none; }
.gradio-container a:hover { color: #00ebb0; }

/* Remove Gradio default group frames — but DON'T break flex rows */
.gradio-container .form,
.gradio-container .panel,
.gradio-container .block.padded {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.gradio-container .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.gradio-container .form { padding: 0 !important; }
.gradio-container > .main { background: transparent !important; }

/* Force horizontal row layout for Gradio Rows tagged pg-row */
.gradio-container .pg-row {
    display: flex !important;
    flex-direction: row !important;
    gap: 12px !important;
    align-items: end !important;
    width: 100% !important;
}
.gradio-container .pg-row > * {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
.gradio-container .pg-row button {
    height: 42px !important;
    align-self: end !important;
    flex-grow: 0 !important;
    flex-basis: auto !important;
    min-width: 120px;
}

/* ── Hero ───────────────────────────────────────────────────── */
.hero {
    background: #141414;
    border: 1px solid #262626;
    border-radius: 10px;
    padding: 14px 20px 12px;
    margin: 8px 0 10px 0;
}
.hero h1 { margin: 0 0 2px 0; font-size: 20px !important; line-height: 1.2; }
.hero .hero-sub {
    color: #9ca3a3;
    font-size: 12px;
    margin: 0 0 10px 0;
    max-width: 720px;
    line-height: 1.45;
}
.hero-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 10px;
}
.stat {
    background: #0a0a0a;
    border: 1px solid #262626;
    border-radius: 8px;
    padding: 10px 14px;
}
.stat .k {
    font-size: 10px;
    color: #9ca3a3;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
    margin-bottom: 3px;
}
.stat .v {
    font-size: 20px;
    color: #f5f5f5;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}

/* ── Section headers ────────────────────────────────────────── */
.section-h {
    margin: 14px 0 2px 0 !important;
    font-size: 17px !important;
}
.section-sub {
    color: #9ca3a3 !important;
    font-size: 12px !important;
    margin: 0 0 10px 0 !important;
}

/* ── Uniqueness cards ───────────────────────────────────────── */
.unique-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 12px;
}
.unique-card {
    background: #141414;
    border: 1px solid #262626;
    border-radius: 10px;
    padding: 18px 20px;
    transition: border-color 0.15s, transform 0.15s;
}
.unique-card:hover {
    border-color: #00d084;
    transform: translateY(-1px);
}
.unique-card .icon-box {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: rgba(0,208,132,0.10);
    border: 1px solid rgba(0,208,132,0.5);
    border-radius: 8px;
    color: #00d084;
    margin-bottom: 12px;
}
.unique-card .icon-box svg { display: block; }
.unique-card h3 {
    font-size: 14px !important;
    font-weight: 600;
    margin: 0 0 4px 0 !important;
    color: #f5f5f5 !important;
}
.unique-card p {
    color: #9ca3a3 !important;
    font-size: 12px !important;
    line-height: 1.55;
    margin: 0 !important;
}

/* ── Tier badges ────────────────────────────────────────────── */
.tier-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border: 1px solid;
    line-height: 1.5;
}
.tier-easy   { color: #00d084; border-color: #00d084; background: rgba(0,208,132,0.08); }
.tier-medium { color: #ffa94d; border-color: #ffa94d; background: rgba(255,169,77,0.08); }
.tier-hard   { color: #ff6b6b; border-color: #ff6b6b; background: rgba(255,107,107,0.08); }

/* ── Scenario cards ─────────────────────────────────────────── */
.scen-card {
    background: #141414;
    border: 1px solid #262626;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.scen-card .scen-id {
    color: #9ca3a3;
    font-size: 11px;
    font-family: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
    margin-left: 10px;
    letter-spacing: 0.2px;
}
.scen-card h2 {
    margin: 10px 0 8px 0 !important;
    font-size: 16px !important;
}
.scen-card .root-cause {
    color: #d1d5d5;
    font-size: 13px;
    line-height: 1.6;
    margin: 10px 0 14px 0;
    padding: 12px 14px;
    background: #0a0a0a;
    border-left: 3px solid #00d084;
    border-radius: 4px;
}
.scen-card .meta {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    margin: 6px 0;
    font-size: 12px;
    color: #9ca3a3;
}
.scen-card .meta b { color: #ededed; font-weight: 600; }
.scen-card .meta code,
.gradio-container code {
    background: #0a0a0a;
    border: 1px solid #262626;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 11px;
    color: #ededed;
    font-family: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
}
.scen-card .graph {
    margin-top: 16px;
    padding: 16px;
    background: #0a0a0a;
    border: 1px solid #262626;
    border-radius: 8px;
    text-align: center;
    min-height: 120px;
}
.scen-card .graph img {
    max-width: 100%;
    height: auto;
    background: transparent;
    display: block;
    margin: 0 auto;
}
.scen-card .graph-caption {
    font-size: 11px;
    color: #9ca3a3;
    margin-top: 8px;
    font-family: "JetBrains Mono", ui-monospace, monospace;
}

/* ── Leaderboard table ──────────────────────────────────────── */
table.lb {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: #141414;
    border: 1px solid #262626;
    border-radius: 10px;
    overflow: hidden;
    font-size: 13px;
}
table.lb th {
    background: #0a0a0a;
    color: #9ca3a3;
    text-align: left;
    padding: 12px 16px;
    font-weight: 600;
    border-bottom: 1px solid #262626;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
table.lb td {
    padding: 10px 16px;
    border-bottom: 1px solid #1a1a1a;
}
table.lb tr:last-child td { border-bottom: none; }
table.lb td.scen-id-cell {
    font-family: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
    font-size: 12px;
    color: #ededed;
}
table.lb td.score {
    text-align: center;
    font-weight: 600;
    color: #0a0a0a;
    font-variant-numeric: tabular-nums;
    font-size: 13px;
}
table.lb tr.tier-avg td {
    background: #0a0a0a;
    color: #ededed;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-top: 1px solid #262626;
}
table.lb tr.tier-avg td.score { color: #0a0a0a; }
table.lb tr.overall td {
    background: #0a0a0a;
    color: #ededed;
    font-weight: 700;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-top: 2px solid #00d084;
}
table.lb tr.overall td.score { color: #0a0a0a; }
.lb-note {
    margin-top: 18px;
    padding: 14px 18px;
    background: rgba(255,107,107,0.06);
    border: 1px solid rgba(255,107,107,0.4);
    border-radius: 8px;
    color: #ffc4c4;
    font-size: 13px;
}
.lb-note b { color: #ff6b6b; }

/* ── Traces ─────────────────────────────────────────────────── */
.trace-header {
    background: #141414;
    border: 1px solid #262626;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 14px;
    display: flex;
    gap: 28px;
    flex-wrap: wrap;
    align-items: center;
}
.trace-header .kv .k {
    color: #9ca3a3;
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.8px;
}
.trace-header .kv .v {
    color: #f5f5f5;
    font-weight: 600;
    font-size: 14px;
}
.trace-header .final {
    margin-left: auto;
    padding: 6px 18px;
    border-radius: 999px;
    color: #0a0a0a;
    font-weight: 700;
    font-size: 14px;
    font-variant-numeric: tabular-nums;
}
.trace-step {
    background: #141414;
    border-left: 3px solid #262626;
    padding: 8px 14px;
    margin: 4px 0;
    font-family: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
    font-size: 12px;
    border-radius: 0 6px 6px 0;
}
.trace-step.progress  { border-left-color: #00d084; }
.trace-step.recovery  { border-left-color: #00d084; background: rgba(0,208,132,0.06); }
.trace-step.worsened  { border-left-color: #ff6b6b; background: rgba(255,107,107,0.06); }
.trace-step.no_effect { border-left-color: #5a5a5a; }
.trace-step .sn { color: #5a5a5a; }
.trace-step .tn { color: #00d084; font-weight: 600; }
.trace-step .ta { color: #9ca3a3; }
.trace-step .oc { font-weight: 600; margin-left: 6px; font-size: 11px; }
.trace-step .oc.progress, .trace-step .oc.recovery { color: #00d084; }
.trace-step .oc.worsened  { color: #ff6b6b; }
.trace-step .oc.no_effect { color: #9ca3a3; }
.trace-step .msg {
    color: #9ca3a3;
    margin-top: 4px;
    font-size: 11px;
    line-height: 1.55;
    font-family: "Inter", -apple-system, sans-serif;
}
details.trace-group {
    background: #141414;
    border-left: 3px solid #262626;
    border-radius: 0 6px 6px 0;
    margin: 4px 0;
}
details.trace-group summary {
    padding: 8px 14px;
    cursor: pointer;
    color: #9ca3a3;
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    user-select: none;
    list-style: none;
}
details.trace-group summary::-webkit-details-marker { display: none; }
details.trace-group summary::before {
    content: "▸ ";
    color: #00d084;
}
details.trace-group[open] summary::before { content: "▾ "; }
details.trace-group summary:hover { color: #ededed; }
details.trace-group > div { padding: 4px 12px 6px 20px; }

/* ── Playground ─────────────────────────────────────────────── */
.pg-alert {
    background: #141414;
    border: 1px solid #00d084;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 14px;
    white-space: pre-wrap;
    font-size: 13px;
    color: #ededed;
    line-height: 1.55;
}
.pg-obs {
    background: #0a0a0a;
    border: 1px solid #262626;
    border-radius: 8px;
    padding: 14px 18px;
    font-family: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
    font-size: 12px;
    color: #ededed;
    max-height: 440px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.55;
}
.pg-status {
    display: flex;
    gap: 22px;
    flex-wrap: wrap;
    margin-bottom: 12px;
    font-size: 13px;
    background: #141414;
    border: 1px solid #262626;
    border-radius: 8px;
    padding: 12px 18px;
}
.pg-status .k {
    color: #9ca3a3;
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.8px;
}
.pg-status .v {
    color: #f5f5f5;
    font-weight: 600;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
}
.pg-status .state-ok   { color: #00d084; }
.pg-status .state-bad  { color: #ff6b6b; }
.pg-status .state-warn { color: #ffa94d; }

/* ── Code blocks ────────────────────────────────────────────── */
.gradio-container pre {
    background: #0a0a0a !important;
    color: #ededed !important;
    font-family: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace !important;
    border: 1px solid #262626;
    border-radius: 8px;
    padding: 16px 18px !important;
    line-height: 1.55;
    font-size: 12px;
    overflow-x: auto;
}
.gradio-container pre code {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    font-size: 12px !important;
}

/* ── Gradio component overrides ─────────────────────────────── */

/* Tabs — force our accent */
.tab-nav, [class*="tab-nav"], .tabs > .tab-nav {
    border-bottom: 1px solid #262626 !important;
    background: transparent !important;
    margin-bottom: 18px !important;
}
.tab-nav button,
.tabs button[role="tab"],
button[role="tab"] {
    color: #9ca3a3 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    margin-right: 4px !important;
}
.tab-nav button:hover,
button[role="tab"]:hover {
    color: #ededed !important;
}
.tab-nav button.selected,
.tab-nav button[aria-selected="true"],
button[role="tab"].selected,
button[role="tab"][aria-selected="true"] {
    color: #00d084 !important;
    border-bottom-color: #00d084 !important;
    background: transparent !important;
}

/* Buttons */
.gradio-container button.primary {
    background: #00d084 !important;
    color: #0a0a0a !important;
    border: 1px solid #00a866 !important;
    font-weight: 600 !important;
    transition: background 0.15s;
}
.gradio-container button.primary:hover {
    background: #00ebb0 !important;
    border-color: #00d084 !important;
}
.gradio-container button.secondary {
    background: #141414 !important;
    color: #ededed !important;
    border: 1px solid #262626 !important;
}
.gradio-container button.secondary:hover {
    border-color: #00d084 !important;
}

/* Inputs, textareas, dropdowns */
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap-inner {
    background: #0a0a0a !important;
    color: #ededed !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
    font-family: "Inter", -apple-system, sans-serif !important;
    font-size: 13px !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus,
.gradio-container .wrap-inner:focus-within {
    border-color: #00d084 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(0,208,132,0.15) !important;
}

/* Dropdown — kill default focus ring / selected background / inner box */
.gradio-container .wrap.svelte-1sk0pyu,
.gradio-container [data-testid="dropdown"],
.gradio-container .wrap-inner {
    background: #0a0a0a !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
}
.gradio-container input[role="listbox"],
.gradio-container input[role="combobox"],
.gradio-container .wrap.svelte-1sk0pyu input,
.gradio-container [data-testid="dropdown"] input,
.gradio-container .wrap-inner input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: #ededed !important;
}
/* Nested container around the dropdown — kill its own border */
.gradio-container .block > .wrap,
.gradio-container .block > .wrap-inner,
.gradio-container .form > .block,
.gradio-container .form > div > .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.gradio-container ul.options,
.gradio-container [role="listbox"] {
    background: #141414 !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
}
.gradio-container ul.options li,
.gradio-container [role="option"] {
    color: #ededed !important;
    padding: 8px 12px !important;
}
.gradio-container ul.options li:hover,
.gradio-container [role="option"]:hover,
.gradio-container [role="option"][aria-selected="true"] {
    background: rgba(0,208,132,0.15) !important;
    color: #00d084 !important;
}

/* Labels */
.gradio-container label, .gradio-container .label-wrap > span {
    color: #9ca3a3 !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Footer ─────────────────────────────────────────────────── */
.footer {
    text-align: center;
    color: #5a5a5a;
    font-size: 11px;
    margin-top: 48px;
    padding: 20px 0 0 0;
    border-top: 1px solid #262626;
}
.footer a { color: #00d084; }
"""


# ── Playground session management ────────────────────────────────

_playground_sessions: Dict[str, SREIncidentEnvironment] = {}
_playground_lock = threading.Lock()


def _pg_reset(scenario_id: str) -> Tuple[str, str, str, str]:
    """Create a fresh env instance, reset with the given scenario, return displays."""
    session_id = str(uuid.uuid4())
    env = SREIncidentEnvironment()
    obs = env.reset(scenario_id=scenario_id, difficulty="medium")
    with _playground_lock:
        _playground_sessions[session_id] = env
    alert = (obs.metadata or {}).get("message", "") if obs.metadata else ""
    status = _pg_status_html(env)
    obs_html = f"<div class='pg-alert'>{_esc(alert)}</div>"
    return session_id, status, obs_html, scenario_id


def _pg_status_html(env: Optional[SREIncidentEnvironment]) -> str:
    if env is None or env._state_machine is None:
        return "<div class='pg-status'><div><div class='k'>Status</div><div class='v'>Not started</div></div></div>"
    sm = env._state_machine
    state = sm.system_state
    healthy = sm.is_resolved()
    state_cls = "state-ok" if healthy else ("state-bad" if state.startswith(("critical", "etcd_crashed", "broken")) else "state-warn")
    reward_display = f"{env._current_reward:.4f}" if env._done else "—"
    return f"""
<div class='pg-status'>
  <div><div class='k'>Scenario</div><div class='v'>{env._scenario['id'] if env._scenario else '—'}</div></div>
  <div><div class='k'>State</div><div class='v {state_cls}'>{state}</div></div>
  <div><div class='k'>Steps used</div><div class='v'>{env._queries_used}</div></div>
  <div><div class='k'>Harm events</div><div class='v'>{len(sm.harm_events)}</div></div>
  <div><div class='k'>Done</div><div class='v'>{env._done}</div></div>
  <div><div class='k'>Final reward</div><div class='v'>{reward_display}</div></div>
</div>
"""


def _pg_call_tool(
    session_id: str,
    tool_name: str,
    service: str,
    action: str,
    params_json: str,
    level_filter: str,
    metric: str,
) -> Tuple[str, str]:
    """Call a tool on the active playground env."""
    if not session_id:
        return (
            "<div class='pg-status'><div><div class='k'>Error</div><div class='v state-bad'>No active session. Click Reset first.</div></div></div>",
            "<div class='pg-obs'>No active session. Click Reset first.</div>",
        )
    with _playground_lock:
        env = _playground_sessions.get(session_id)
    if env is None:
        return (
            "<div class='pg-status'><div><div class='k'>Error</div><div class='v state-bad'>Session expired. Reset again.</div></div></div>",
            "<div class='pg-obs'>Session expired. Click Reset to start a new one.</div>",
        )
    if env._done:
        return _pg_status_html(env), "<div class='pg-obs'>Episode is done. Reset to start a new one.</div>"

    if get_server_tools is None:
        return _pg_status_html(env), "<div class='pg-obs'>Error: openenv MCP tooling unavailable.</div>"

    server_tools = get_server_tools(env.mcp_server)
    if tool_name not in server_tools:
        return _pg_status_html(env), f"<div class='pg-obs'>Error: tool '{tool_name}' not found.</div>"

    # Build kwargs based on tool
    kwargs: Dict[str, Any] = {}
    if tool_name == "list_services":
        pass
    elif tool_name == "read_logs":
        kwargs = {"service": service, "window_minutes": 10}
        if level_filter and level_filter != "any":
            kwargs["level_filter"] = level_filter
    elif tool_name == "check_metric":
        kwargs = {"service": service, "metric": metric or "error_rate", "window_minutes": 10}
    elif tool_name == "get_service_info":
        kwargs = {"service": service}
    elif tool_name in ("restart_service", "rollback_deploy"):
        kwargs = {"service": service}
    elif tool_name == "scale_replicas":
        try:
            count = int(params_json) if params_json.strip() else 2
        except ValueError:
            count = 2
        kwargs = {"service": service, "count": count}
    elif tool_name == "execute_runbook":
        try:
            params = params_json.strip() if params_json.strip() else None
        except Exception:
            params = None
        kwargs = {"service": service, "action": action, "params": params}
    elif tool_name == "verify_resolution":
        try:
            parsed = json.loads(params_json) if params_json.strip() else {}
        except json.JSONDecodeError:
            parsed = {}
        kwargs = {
            "affected_service": parsed.get("affected_service", service),
            "failure_type": parsed.get("failure_type", "unknown"),
            "root_cause": parsed.get("root_cause", action or "diagnosis not provided"),
        }

    tool = server_tools[tool_name]
    try:
        if _inspect.iscoroutinefunction(tool.fn):
            import asyncio as _asyncio
            result = _asyncio.run(tool.fn(**kwargs))
        else:
            result = tool.fn(**kwargs)
    except Exception as e:
        result = json.dumps({"error": str(e)})

    env._steps += 1
    if env._state is not None:
        env._state.step_count = env._steps

    # Format result
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
        result_display = json.dumps(parsed, indent=2)
    except (json.JSONDecodeError, TypeError):
        result_display = str(result)

    call_label = f"▸ {tool_name}({json.dumps(kwargs, separators=(',', ':'))})"
    obs_html = f"<div class='pg-obs'>{_esc(call_label)}\n\n{_esc(result_display[:4000])}</div>"
    return _pg_status_html(env), obs_html


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── HTML builders ─────────────────────────────────────────────────


def _hero_html() -> str:
    scenarios = load_scenarios()
    total_states = sum(len(s["states"]) for s in scenarios)
    total_actions = sum(
        sum(len(st.get("actions", [])) for st in s["states"].values())
        for s in scenarios
    )
    return f"""
<div class="hero">
  <h1>SRE Incident Response Environment</h1>
  <p class="hero-sub">A state-graph maze where LLM agents investigate, diagnose, and
  resolve production incidents across a full SRE stack — from kernel networking
  to distributed consensus to TLS/PKI. Scenarios based on real post-mortems.</p>
  <div class="hero-stats">
    <div class="stat"><div class="k">Scenarios</div><div class="v">{len(scenarios)}</div></div>
    <div class="stat"><div class="k">State Nodes</div><div class="v">{total_states}</div></div>
    <div class="stat"><div class="k">Actions</div><div class="v">{total_actions}</div></div>
    <div class="stat"><div class="k">Reward Comps</div><div class="v">7</div></div>
    <div class="stat"><div class="k">Tiers</div><div class="v">3</div></div>
    <div class="stat"><div class="k">Baselined</div><div class="v">3 LLMs</div></div>
  </div>
</div>
"""


# Lucide-style line-art icons, 20x20, currentColor stroke
ICON_SVG = {
    "zap":       '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "layers":    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "terminal":  '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>',
    "network":   '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M6 21V9a9 9 0 0 0 9 9"/></svg>',
    "route":     '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/></svg>',
    "trend":     '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>',
}

UNIQUENESS = [
    ("zap", "Real production incident patterns",
     "Scenarios drawn from actual post-mortems: kernel TCP rmem_max silent drops, "
     "CPU microcode TSC drift destabilizing Raft consensus, JVM classloader "
     "metaspace leaks, NUMA cross-socket migration, WAL archiver disk exhaustion, "
     "etcd quota alarms, Kafka/Zookeeper rebalance storms, mTLS cert expiry."),
    ("layers", "Full SRE-stack breadth",
     "Kernel networking · CPU/hardware · JVM internals · Postgres WAL · etcd/Raft · "
     "Kafka/Zookeeper · TLS/PKI · Kubernetes API server. The full layer stack an "
     "on-call engineer actually sees."),
    ("terminal", "Realistic SRE tool interface",
     "Nine MCP tools mirror a real on-call toolkit: list_services, read_logs, "
     "check_metric, get_service_info, restart_service, rollback_deploy, "
     "scale_replicas, execute_runbook, verify_resolution."),
    ("network", "Multi-step cross-service remediation",
     "Every scenario requires 4–5 sequential correct actions across multiple "
     "services — root-cause service → affected services → cleanup → prevention. "
     "Wrong actions trigger worsened state transitions that move the system "
     "backward, mirroring real production."),
    ("route", "State-graph maze with trap actions",
     "Each scenario is a directed graph with progress / no_effect / worsened / "
     "recovery transitions. Partial credit is awarded quadratically based on BFS "
     "depth along the optimal path — not binary fixed/not-fixed."),
    ("trend", "Frontier-model difficulty gradient verified",
     "The hardest scenario (wal_archive_disk_full_h002) scores 0.04 average "
     "across GPT-5.4, o4-mini, and GPT-4o-mini — genuinely floors frontier "
     "models and leaves meaningful headroom for better agents."),
]


def _uniqueness_html() -> str:
    cards = []
    for icon_key, title, body in UNIQUENESS:
        icon = ICON_SVG.get(icon_key, "")
        cards.append(f"""
<div class="unique-card">
  <div class="icon-box">{icon}</div>
  <h3>{title}</h3>
  <p>{body}</p>
</div>""")
    return f'<div class="unique-grid">{"".join(cards)}</div>'


def _scenarios_html() -> str:
    scenarios = load_scenarios()
    order = {"easy": 0, "medium": 1, "hard": 2}
    scenarios.sort(key=lambda s: (order.get(s["difficulty"], 99), s["id"]))
    cards = []
    for s in scenarios:
        tier = s["difficulty"]
        # Pre-generated SVG served from /state_graphs/ static mount
        svg_url = f"/state_graphs/{s['id']}.svg"
        services = ", ".join(f"<code>{sv}</code>" for sv in s["services"][:8])
        if len(s["services"]) > 8:
            services += f" <span style='color:#6e7681'>+{len(s['services'])-8}</span>"
        chain = " → ".join(s["causal_chain"]) if s["causal_chain"] else "—"
        cards.append(f"""
<div class="scen-card">
  <span class="tier-badge tier-{tier}">{tier}</span>
  <span class="scen-id">{s["id"]}</span>
  <h2>{_esc(s["title"])}</h2>
  <div class="root-cause">{_esc(s["root_cause"])}</div>
  <div class="meta">
    <span><b>root_service:</b> <code>{s["root_service"]}</code></span>
    <span><b>optimal_steps:</b> {s["optimal_steps"]}</span>
    <span><b>state_nodes:</b> {len(s["states"])}</span>
    <span><b>duration:</b> {s["duration_minutes"]} min</span>
  </div>
  <div class="meta"><span><b>services:</b> {services}</span></div>
  <div class="meta"><span><b>causal_chain:</b> {chain}</span></div>
  <div class="graph">
    <img src="{svg_url}" alt="State graph for {s['id']}" loading="lazy"/>
    <div class="graph-caption">green = correct progress path · red dashed = wrong actions (harm pool) · ⚠ red node = distinct trap state</div>
  </div>
</div>""")
    return "\n".join(cards)


def _leaderboard_html() -> str:
    scenarios = load_scenarios()
    order = {"easy": 0, "medium": 1, "hard": 2}
    scenarios.sort(key=lambda s: (order.get(s["difficulty"], 99), s["id"]))
    lb = leaderboard_averages()

    rows: List[str] = ["<table class='lb'>"]
    rows.append(
        "<thead><tr><th>Scenario</th><th>Tier</th>"
        + "".join(f"<th>{m}</th>" for m in MODELS)
        + "</tr></thead><tbody>"
    )

    tier_buckets: Dict[str, Dict[str, List[float]]] = {
        m: {"easy": [], "medium": [], "hard": []} for m in MODELS
    }

    for s in scenarios:
        tier = s["difficulty"]
        row = f"<tr><td class='scen-id-cell'>{s['id']}</td>"
        row += f"<td><span class='tier-badge tier-{tier}'>{tier}</span></td>"
        for m in MODELS:
            avg = lb.get(m, {}).get("per_scenario", {}).get(s["id"])
            if avg is None:
                row += "<td class='score' style='background:#21262d;color:#6e7681'>—</td>"
            else:
                row += f"<td class='score' style='background:{score_color(avg)}'>{avg:.3f}</td>"
                tier_buckets[m][tier].append(avg)
        row += "</tr>"
        rows.append(row)

    for t in ["easy", "medium", "hard"]:
        rows.append("<tr class='tier-avg'>")
        rows.append(f"<td colspan='2'>{t.upper()} TIER AVG</td>")
        for m in MODELS:
            vals = tier_buckets[m][t]
            avg = sum(vals) / len(vals) if vals else 0.0
            rows.append(f"<td class='score' style='background:{score_color(avg)}'>{avg:.3f}</td>")
        rows.append("</tr>")

    rows.append("<tr class='overall'>")
    rows.append("<td colspan='2'>OVERALL</td>")
    for m in MODELS:
        overall = lb.get(m, {}).get("overall", 0.0)
        rows.append(f"<td class='score' style='background:{score_color(overall)}'>{overall:.3f}</td>")
    rows.append("</tr>")
    rows.append("</tbody></table>")

    rows.append(
        "<div class='lb-note'>"
        "<b>Anti-saturation evidence:</b> "
        "<code>wal_archive_disk_full_h002</code> scores <b>0.04 avg</b> across all "
        "3 frontier models. No model has solved it. This scenario genuinely "
        "challenges the ceiling."
        "</div>"
    )
    return "\n".join(rows)


def _trace_html(model: Optional[str], scenario_id: Optional[str]) -> str:
    if not model or not scenario_id:
        return "<div class='pg-obs'>Select a model and scenario to view the trace.</div>"
    raw = load_trace(model, scenario_id)
    if raw is None:
        return (
            f"<div class='lb-note'>No trace available for "
            f"<code>{model}</code> × <code>{scenario_id}</code>. "
            f"This episode may have failed or timed out.</div>"
        )
    summary = summarize_trace(raw)
    final = summary["reward"]
    if final is None:
        bg = "#5a5a5a"
        final_label = "—"
        verdict_html = ""
    else:
        bg = score_color(final)
        final_label = f"{final:.4f}"
        # Heuristic: a "solved" episode requires reward ≥ 0.50 (exit + diagnosis
        # are gated on system_healthy, so anything ≥ 0.50 reached healthy)
        if final >= 0.50:
            verdict_html = (
                "<div class='final' style='background:#00d084;margin-left:0;"
                "margin-right:8px'>SOLVED</div>"
            )
        else:
            verdict_html = (
                "<div class='final' style='background:#ff6b6b;margin-left:0;"
                "margin-right:8px'>NOT SOLVED</div>"
            )
    steps_used = summary.get("steps_used") if summary.get("steps_used") is not None else "—"
    elapsed = summary.get("elapsed_seconds") if summary.get("elapsed_seconds") is not None else "—"
    error_html = ""
    if summary.get("error"):
        error_html = (
            "<div class='lb-note' style='margin-bottom:12px'>"
            f"<b>Episode error:</b> {_esc(str(summary['error'])[:280])}"
            "</div>"
        )
    header = f"""
{error_html}
<div class="trace-header">
  <div class="kv"><div class="k">Model</div><div class="v">{summary['model']}</div></div>
  <div class="kv"><div class="k">Scenario</div><div class="v">{summary['scenario_id']}</div></div>
  <div class="kv"><div class="k">Steps</div><div class="v">{steps_used}</div></div>
  <div class="kv"><div class="k">Elapsed</div><div class="v">{elapsed}s</div></div>
  <div style="margin-left:auto;display:flex;gap:8px;align-items:center">
    {verdict_html}
    <div class="final" style="background:{bg};margin-left:0">{final_label}</div>
  </div>
</div>
"""

    INVESTIGATION = {"list_services", "read_logs", "check_metric", "get_service_info"}
    step_htmls: List[str] = []
    i = 0
    steps = summary["steps"]
    while i < len(steps):
        step = steps[i]
        if step["tool"] in INVESTIGATION and not step["outcome"]:
            start = i
            while i < len(steps) and steps[i]["tool"] in INVESTIGATION and not steps[i]["outcome"]:
                i += 1
            run = steps[start:i]
            if len(run) <= 2:
                for st in run:
                    step_htmls.append(_render_step(st))
            else:
                inner = "\n".join(_render_step(st) for st in run)
                step_htmls.append(f"""
<details class="trace-group">
  <summary>T{run[0]['n']:03d}–T{run[-1]['n']:03d} · {len(run)} investigation queries (click to expand)</summary>
  <div>{inner}</div>
</details>""")
        else:
            step_htmls.append(_render_step(step))
            i += 1
    return header + "\n".join(step_htmls)


def _render_step(step: Dict[str, Any]) -> str:
    outcome = step.get("outcome") or ""
    css_class = outcome or "investigate"
    emoji = OUTCOME_EMOJI.get(outcome, "·") if outcome else "·"
    label = outcome.upper() if outcome else ""
    args = json.dumps(step["args"], separators=(",", ":"))
    if len(args) > 180:
        args = args[:177] + "…"
    args = _esc(args)
    msg = _esc(step.get("message") or "")
    oc = f'<span class="oc {outcome}">[{emoji} {label}]</span>' if outcome else ""
    msg_html = f'<div class="msg">↳ {msg}</div>' if msg else ""
    return f"""
<div class="trace-step {css_class}">
  <span class="sn">T{step['n']:03d}</span>
  <span class="tn">{step['tool']}</span>
  <span class="ta">({args})</span>
  {oc}
  {msg_html}
</div>"""


def _try_it_html() -> str:
    return """
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:22px 26px">
  <h3 style="color:#f0f6fc;margin-top:0">Run inference.py</h3>
  <p style="color:#8b949e">Point any OpenAI-compatible LLM at the deployed HF Space:</p>
  <pre><code>export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-5.4"
export HF_TOKEN="$OPENAI_API_KEY"

# Default: 1 run per scenario × 8 scenarios = 8 episodes (~8 min)
python inference.py --space https://maverick98-sre-incident-env.hf.space

# 2 runs per scenario for variance estimates
python inference.py --space https://maverick98-sre-incident-env.hf.space --episodes 2

# Single tier
python inference.py --space https://maverick98-sre-incident-env.hf.space --difficulty hard</code></pre>

  <h3 style="color:#f0f6fc;margin-top:20px">Structured output format</h3>
  <p style="color:#8b949e">Inference emits <code>[START] / [STEP] / [END]</code> blocks on stdout:</p>
  <pre><code>[START] task=jvm_metaspace_classloader_leak_001
[STEP] step=1 reward=0.0001
[STEP] step=2 reward=0.0001
...
[STEP] step=32 reward=0.8917
[END] task=jvm_metaspace_classloader_leak_001 score=0.8917 steps=32</code></pre>

  <h3 style="color:#f0f6fc;margin-top:20px">Links</h3>
  <ul style="color:#c9d1d9">
    <li>GitHub · <a href="https://github.com/Mohit-Dhawan98/sre-incident-env" style="color:#58a6ff">Mohit-Dhawan98/sre-incident-env</a></li>
    <li>HF Space · <a href="https://huggingface.co/spaces/Maverick98/sre-incident-env" style="color:#58a6ff">Maverick98/sre-incident-env</a></li>
    <li>Dataset · <code>scenarios/incidents_v3.jsonl</code> · 8 scenarios · 50 states · 159 actions</li>
  </ul>
</div>
"""


# ── App factory ────────────────────────────────────────────────────


def create_landing_app() -> gr.Blocks:
    """Build the Gradio Blocks app that gets mounted at '/'."""

    scenarios = load_scenarios()
    scenario_choices = [s["id"] for s in scenarios]
    tool_names = [
        "list_services",
        "read_logs",
        "check_metric",
        "get_service_info",
        "restart_service",
        "rollback_deploy",
        "scale_replicas",
        "execute_runbook",
        "verify_resolution",
    ]

    with gr.Blocks(title="SRE Incident Response Environment") as app:
        # Load Inter + JetBrains Mono from Google Fonts
        gr.HTML(
            '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&'
            'family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">'
        )
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.HTML(_hero_html())

        with gr.Tabs():

            with gr.Tab("Overview"):
                gr.HTML("<h2 class='section-h'>What makes this unique</h2>")
                gr.HTML("<p class='section-sub'>Six design decisions that differentiate this environment.</p>")
                gr.HTML(_uniqueness_html())

            with gr.Tab("Scenarios"):
                gr.HTML("<h2 class='section-h'>8 production incidents · 3 tiers</h2>")
                gr.HTML(
                    "<p class='section-sub'>Each scenario is a state-graph maze. "
                    "Click <b>View state graph</b> to see the remediation topology.</p>"
                )
                gr.HTML(_scenarios_html())

            with gr.Tab("Leaderboard"):
                gr.HTML("<h2 class='section-h'>Frontier model baselines</h2>")
                gr.HTML(
                    "<p class='section-sub'>Scores from 16 episodes per model "
                    "(8 scenarios × 2 runs) against this HF Space. Each cell is "
                    "the per-scenario average.</p>"
                )
                gr.HTML(_leaderboard_html())

            with gr.Tab("Traces"):
                gr.HTML("<h2 class='section-h'>Agent trace viewer</h2>")
                gr.HTML(
                    "<p class='section-sub'>Step-by-step replay of what each model "
                    "did on each scenario. Investigation steps are collapsed by "
                    "default — click to expand.</p>"
                )
                with gr.Row():
                    model_dd = gr.Dropdown(
                        choices=MODELS,
                        value=MODELS[0],
                        label="Model",
                        interactive=True,
                    )
                    scenario_dd = gr.Dropdown(
                        choices=scenario_choices,
                        value=scenario_choices[0],
                        label="Scenario",
                        interactive=True,
                    )
                trace_out = gr.HTML(_trace_html(MODELS[0], scenario_choices[0]))

                def _update_trace(model: str, sid: str) -> str:
                    return _trace_html(model, sid)

                model_dd.change(_update_trace, inputs=[model_dd, scenario_dd], outputs=trace_out)
                scenario_dd.change(_update_trace, inputs=[model_dd, scenario_dd], outputs=trace_out)

            with gr.Tab("Playground"):
                gr.HTML("<h2 class='section-h'>Live interactive episode</h2>")
                gr.HTML(
                    "<p class='section-sub'>Pick a scenario, reset the environment, then "
                    "call tools step by step against the real state machine. Watch the "
                    "system state change after each action.</p>"
                )

                session_state = gr.State(value="")
                with gr.Row(elem_classes="pg-row", equal_height=True):
                    pg_scenario = gr.Dropdown(
                        choices=scenario_choices,
                        value=scenario_choices[0],
                        label="Scenario",
                        scale=5,
                    )
                    pg_reset_btn = gr.Button("Reset episode", variant="primary", scale=1, min_width=140)

                pg_status_out = gr.HTML(_pg_status_html(None))
                pg_obs_out = gr.HTML("<div class='pg-obs'>Click Reset to start an episode.</div>")

                gr.HTML("<h3 class='section-h' style='margin-top:22px'>Call a tool</h3>")
                with gr.Row(elem_classes="pg-row", equal_height=True):
                    pg_tool = gr.Dropdown(
                        choices=tool_names,
                        value="list_services",
                        label="Tool",
                        scale=1,
                    )
                    pg_service = gr.Textbox(
                        label="Service",
                        value="",
                        scale=1,
                        placeholder="e.g. zookeeper",
                    )
                with gr.Row(elem_classes="pg-row", equal_height=True):
                    pg_action = gr.Textbox(
                        label="Action (execute_runbook)",
                        value="",
                        scale=1,
                        placeholder="e.g. update_config",
                    )
                    pg_params = gr.Textbox(
                        label="Params / count / verify (JSON)",
                        value="",
                        scale=1,
                        placeholder='e.g. {"key":"session_timeout_ms","value":"30000"}',
                    )
                with gr.Row(elem_classes="pg-row", equal_height=True):
                    pg_level = gr.Dropdown(
                        choices=["any", "ERROR", "WARN", "INFO"],
                        value="any",
                        label="Log level",
                        scale=1,
                    )
                    pg_metric = gr.Textbox(
                        label="Metric",
                        value="error_rate",
                        scale=1,
                    )
                    pg_step_btn = gr.Button("Step", variant="primary", scale=1, min_width=140)

                def _reset_click(scenario_id: str):
                    sid, status, obs, _ = _pg_reset(scenario_id)
                    return sid, status, obs

                pg_reset_btn.click(
                    _reset_click,
                    inputs=[pg_scenario],
                    outputs=[session_state, pg_status_out, pg_obs_out],
                )

                pg_step_btn.click(
                    _pg_call_tool,
                    inputs=[
                        session_state, pg_tool, pg_service, pg_action,
                        pg_params, pg_level, pg_metric,
                    ],
                    outputs=[pg_status_out, pg_obs_out],
                )

            with gr.Tab("Try It"):
                gr.HTML("<h2 class='section-h'>Run the baseline yourself</h2>")
                gr.HTML(_try_it_html())

        gr.HTML("""
<div class="footer">
  sre-incident-env · 7-component reward · 8 scenarios · 50 states · 159 actions · deterministic grader<br/>
  <a href="https://github.com/Mohit-Dhawan98/sre-incident-env">github</a> ·
  <a href="https://huggingface.co/spaces/Maverick98/sre-incident-env">huggingface</a>
</div>
""")

    return app
