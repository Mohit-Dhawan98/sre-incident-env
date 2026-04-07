#!/usr/bin/env python3
"""Test scenario(s) with full trace logging.

Saves complete trace per scenario to outputs/ directory.
All config via CLI args.

Usage:
    # Single scenario
    python scripts/test_scenario.py --scenario kafka_partition_rebalance_storm_001 --difficulty medium

    # Multiple scenarios
    python scripts/test_scenario.py \\
        --scenario cert_expiry_mutual_tls_001 \\
        --scenario kernel_tcp_rmem_silent_drop_001 \\
        --difficulty expert

    # All scenarios from a tier
    python scripts/test_scenario.py --tier hard

    # Custom config
    python scripts/test_scenario.py \\
        --tier medium \\
        --model gpt-5.4 \\
        --api-base https://api.openai.com/v1 \\
        --api-key $OPENAI_API_KEY \\
        --scenario-file scenarios/incidents_v3.jsonl \\
        --server-url http://127.0.0.1:8766 \\
        --max-steps 200 \\
        --runs 2 \\
        --output-dir outputs/manual_test
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from client import SREIncidentEnvHTTP

SYSTEM_PROMPT = """You are an expert on-call Site Reliability Engineer responding to a production incident.

# MISSION
Investigate the incident, identify the root cause, FIX the problem, and verify resolution.
This is a LIVE system — your actions have real consequences. Wrong fixes can make things worse.

# PROTOCOL
1. ORIENT: list_services to see the topology.
2. INVESTIGATE: read_logs across services to find which are affected and trace the error chain upstream.
3. DISCOVER: get_service_info on suspected services to learn what actions, config params, and recent deploys are available.
4. REMEDIATE: Apply a fix based on your diagnosis.
5. OBSERVE: After ANY remediation, call read_logs to see what changed.
6. VERIFY: When the system is healthy, call verify_resolution with your diagnosis.

# CRITICAL RULES
- DISCOVER BEFORE FIXING: Call get_service_info before using execute_runbook.
- OBSERVE AFTER FIXING: Always read_logs after a remediation to check the outcome.
- CAUSE ≠ EFFECT: The service with the most errors is usually a VICTIM, not the cause.
"""


def mcp_tools_to_openai(tools_raw):
    out = []
    for t in tools_raw:
        schema = t.get("inputSchema", {}) if isinstance(t, dict) else getattr(t, "input_schema", {}) or {}
        properties = {}
        required = []
        if schema and "properties" in schema:
            for n, sc in schema["properties"].items():
                prop = {"type": sc.get("type", "string")}
                if "description" in sc:
                    prop["description"] = sc["description"]
                properties[n] = prop
            required = schema.get("required", [])
        name = t["name"] if isinstance(t, dict) else t.name
        desc = t.get("description", "") if isinstance(t, dict) else (t.description or "")
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        })
    return out


async def run_scenario(
    llm: OpenAI,
    server_url: str,
    scenario_id: str,
    difficulty: str,
    model: str,
    max_steps: int,
    run_num: int,
) -> Dict[str, Any]:
    """Run one scenario, return full trace + summary."""
    trace = []
    started = time.time()
    is_reasoning = any(x in model for x in ["o3", "o4", "gpt-5"])

    try:
        async with SREIncidentEnvHTTP(base_url=server_url, timeout=120) as env:
            # Reset
            obs = await env.reset(difficulty=difficulty, scenario_id=scenario_id)
            tools_raw = await env.list_tools()
            tools = mcp_tools_to_openai(tools_raw)

            alert = obs.get("message", "Production incident detected.") if isinstance(obs, dict) else "Production incident."
            trace.append({"role": "system", "content": SYSTEM_PROMPT})
            trace.append({"role": "user", "content": alert})

            messages = list(trace)
            done = False
            final_reward = None
            steps_taken = 0
            error = None

            for step in range(1, max_steps + 1):
                steps_taken = step
                create_kwargs = {
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                }
                if is_reasoning:
                    create_kwargs["max_completion_tokens"] = 2000
                else:
                    create_kwargs["temperature"] = 0.1
                    create_kwargs["max_tokens"] = 500

                try:
                    resp = llm.chat.completions.create(**create_kwargs, timeout=90)
                except Exception as e:
                    error = f"API error: {str(e)[:200]}"
                    print(f"  T{step}: API ERROR - {error[:80]}", flush=True)
                    break

                msg = resp.choices[0].message

                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    args_str = json.dumps(tool_args)
                    print(f"  T{step}: {tool_name}({args_str[:120]})", end="", flush=True)

                    asst_msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tc.function.arguments},
                        }],
                    }
                    messages.append(asst_msg)
                    trace.append({**asst_msg, "_step": step})

                    try:
                        result = await env.call_tool(tool_name, **tool_args)
                    except Exception as e:
                        result = json.dumps({"error": f"transport_error: {str(e)[:100]}"})

                    if not isinstance(result, str):
                        result = json.dumps(result)

                    # Extract outcome for inline display
                    outcome = None
                    try:
                        parsed = json.loads(result)
                        outcome = parsed.get("outcome", "")
                    except json.JSONDecodeError:
                        pass

                    if outcome:
                        print(f" [{outcome}]", end="")

                    if env._last_done:
                        final_reward = env._last_reward
                        print(f" → reward={final_reward:.4f}", flush=True)
                        done = True
                        tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
                        messages.append(tool_msg)
                        trace.append({**tool_msg, "_step": step})
                        break

                    print(flush=True)
                    tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
                    messages.append(tool_msg)
                    trace.append({**tool_msg, "_step": step})

                elif msg.content:
                    text_msg = {"role": "assistant", "content": msg.content}
                    messages.append(text_msg)
                    trace.append({**text_msg, "_step": step})
                    messages.append({"role": "user", "content": "Please use a tool."})

            elapsed = time.time() - started

            return {
                "summary": {
                    "scenario_id": scenario_id,
                    "difficulty": difficulty,
                    "model": model,
                    "run": run_num,
                    "reward": final_reward,
                    "done": done,
                    "steps": steps_taken,
                    "elapsed_seconds": round(elapsed, 1),
                    "error": error,
                    "max_steps_hit": not done and not error,
                },
                "trace": trace,
            }

    except Exception as e:
        return {
            "summary": {
                "scenario_id": scenario_id,
                "difficulty": difficulty,
                "model": model,
                "run": run_num,
                "reward": None,
                "done": False,
                "steps": 0,
                "elapsed_seconds": round(time.time() - started, 1),
                "error": f"session_error: {str(e)[:200]}",
                "max_steps_hit": False,
            },
            "trace": trace,
        }


def list_scenarios_in_file(scenario_file: str) -> List[Dict[str, str]]:
    """Read scenario file, return list of {id, difficulty}."""
    out = []
    with open(scenario_file) as f:
        for line in f:
            try:
                s = json.loads(line)
                out.append({"id": s["id"], "difficulty": s["difficulty"]})
            except json.JSONDecodeError:
                continue
    return out


async def main_async():
    parser = argparse.ArgumentParser(description="Test SRE incident scenarios with full trace logging")
    parser.add_argument("--scenario", action="append", default=[], help="Scenario ID (repeatable)")
    parser.add_argument("--tier", choices=["easy", "medium", "hard", "expert"], help="Run all scenarios in this tier")
    parser.add_argument("--difficulty", help="Override difficulty for --scenario (auto-detected if omitted)")
    parser.add_argument("--scenario-file", default="scenarios/incidents_v3.jsonl", help="Scenarios JSONL file")
    parser.add_argument("--model", default="gpt-5.4", help="LLM model name")
    parser.add_argument("--api-base", default=os.getenv("API_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN"))
    parser.add_argument("--server-url", default="http://127.0.0.1:8766", help="SRE env server URL")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--runs", type=int, default=1, help="Runs per scenario")
    parser.add_argument("--output-dir", default=None, help="Output dir (default: outputs/test_TIMESTAMP)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: --api-key required (or set OPENAI_API_KEY / HF_TOKEN env var)")
        sys.exit(1)

    # Resolve scenarios
    all_scenarios = list_scenarios_in_file(args.scenario_file)
    targets = []
    if args.tier:
        targets = [s for s in all_scenarios if s["difficulty"] == args.tier]
    if args.scenario:
        for sid in args.scenario:
            match = next((s for s in all_scenarios if s["id"] == sid), None)
            if not match:
                print(f"WARNING: scenario {sid} not found in {args.scenario_file}")
                continue
            if args.difficulty:
                match = {"id": sid, "difficulty": args.difficulty}
            if match not in targets:
                targets.append(match)

    if not targets:
        print("ERROR: No scenarios to run. Use --scenario or --tier")
        sys.exit(1)

    # Setup output dir
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Server: {args.server_url}")
    print(f"Scenario file: {args.scenario_file}")
    print(f"Max steps: {args.max_steps}")
    print(f"Runs per scenario: {args.runs}")
    print(f"Output: {out_dir}")
    print(f"Targets: {len(targets)} scenario(s) x {args.runs} runs = {len(targets) * args.runs} episodes")
    print("=" * 70)

    llm = OpenAI(base_url=args.api_base, api_key=args.api_key)
    all_results = []

    for sc in targets:
        for run_num in range(1, args.runs + 1):
            print(f"\n=== {sc['id']} ({sc['difficulty']}) RUN {run_num} ===")
            result = await run_scenario(
                llm=llm,
                server_url=args.server_url,
                scenario_id=sc["id"],
                difficulty=sc["difficulty"],
                model=args.model,
                max_steps=args.max_steps,
                run_num=run_num,
            )
            all_results.append(result["summary"])

            # Save per-scenario trace
            fname = f"{sc['id']}_run{run_num}.json"
            with open(out_dir / fname, "w") as f:
                json.dump(result, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for r in all_results:
        sid = r["scenario_id"]
        diff = r["difficulty"]
        run = r["run"]
        if r["reward"] is not None:
            status = f"reward={r['reward']:.4f} steps={r['steps']}"
        elif r["max_steps_hit"]:
            status = f"max_steps ({r['steps']})"
        elif r["error"]:
            status = f"error: {r['error'][:50]}"
        else:
            status = "unknown"
        print(f"  [{diff:6s}] {sid[:40]:40s} R{run}: {status}")

    # Save combined
    combined = {
        "model": args.model,
        "scenario_file": args.scenario_file,
        "max_steps": args.max_steps,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }
    with open(out_dir / "_summary.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nFull traces saved to: {out_dir}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
