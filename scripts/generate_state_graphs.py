#!/usr/bin/env python3
"""Pre-generate state-graph SVGs for every scenario.

Calls mermaid.ink once per scenario, validates the response is a real SVG,
writes to outputs/state_graphs/<scenario_id>.svg.

Run this once before deploying:
    uv run python scripts/generate_state_graphs.py

The Gradio UI then serves these files via the /state_graphs/ static mount,
so there's no per-request network call to mermaid.ink at runtime.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import httpx

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.ui_data import build_mermaid, load_scenarios, mermaid_to_url

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "state_graphs"


def fetch_svg(mermaid_code: str) -> str | None:
    """Hit mermaid.ink, return SVG text or None on failure."""
    url = mermaid_to_url(mermaid_code)
    try:
        resp = httpx.get(url, timeout=20.0, follow_redirects=True)
    except Exception as e:
        print(f"  ERROR: HTTP request failed: {e}")
        return None
    if resp.status_code != 200:
        print(f"  ERROR: status {resp.status_code}")
        return None
    text = resp.text
    if not text.lstrip().startswith("<svg"):
        print(f"  ERROR: response doesn't start with <svg (first 80 chars): {text[:80]}")
        return None
    # Drop the @import that some browsers/iframes block
    text = re.sub(r"@import url\([^)]+\);", "", text)
    # Merge our sizing rules into the existing <svg style="..."> attribute,
    # or add a new one if missing — never duplicate.
    extra = "max-width:100%;height:auto;display:block;margin:0 auto"
    if re.search(r'<svg[^>]*\sstyle="', text):
        text = re.sub(
            r'(<svg[^>]*\sstyle=")([^"]*)(")',
            lambda m: f'{m.group(1)}{extra};{m.group(2)}{m.group(3)}',
            text,
            count=1,
        )
    else:
        text = re.sub(
            r"<svg([^>]*)>",
            rf'<svg\1 style="{extra}">',
            text,
            count=1,
        )
    return text


def validate_svg(svg_text: str, scenario_id: str) -> tuple[bool, list[str]]:
    """Sanity checks on a generated SVG.

    Returns (ok, list of issues).
    """
    issues: list[str] = []
    if len(svg_text) < 1000:
        issues.append(f"too small ({len(svg_text)} bytes)")
    if "<svg" not in svg_text:
        issues.append("missing <svg> root tag")
    if "</svg>" not in svg_text:
        issues.append("missing </svg> closing tag")
    # Mermaid renders graph-related elements
    if 'class="flowchart"' not in svg_text and "flowchart" not in svg_text:
        issues.append("no flowchart class — mermaid may have failed to parse")
    # Check that we got at least a few node rects
    rect_count = svg_text.count("<rect")
    if rect_count < 3:
        issues.append(f"only {rect_count} <rect> elements (expected ≥ 3 for state nodes)")
    return (len(issues) == 0, issues)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios()
    print(f"Generating state-graph SVGs for {len(scenarios)} scenarios → {OUT_DIR}")
    print("=" * 70)

    successes = 0
    failures: list[str] = []

    for s in scenarios:
        sid = s["id"]
        print(f"\n[{sid}]")
        mermaid_src = build_mermaid(s)
        print(f"  mermaid LOC: {len(mermaid_src.splitlines())}")

        svg = fetch_svg(mermaid_src)
        if svg is None:
            print(f"  ✗ FAILED to fetch")
            failures.append(sid)
            continue

        ok, issues = validate_svg(svg, sid)
        if not ok:
            print(f"  ✗ VALIDATION FAILED:")
            for issue in issues:
                print(f"      - {issue}")
            failures.append(sid)
            continue

        out_path = OUT_DIR / f"{sid}.svg"
        out_path.write_text(svg)
        print(f"  ✓ wrote {out_path.name} ({len(svg):,} bytes, {svg.count('<rect')} rects, {svg.count('<path')} paths)")
        successes += 1

    print("\n" + "=" * 70)
    print(f"SUMMARY: {successes}/{len(scenarios)} succeeded")
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("All state graphs generated and validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
