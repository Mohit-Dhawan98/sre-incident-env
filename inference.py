#!/usr/bin/env python3
"""Baseline inference for SRE Incident Response environment.

Uses HTTP transport by default (works through HF Space proxy).
Falls back to WebSocket with --websocket flag.

Required env vars:
    API_BASE_URL   — LLM endpoint (default: https://api.openai.com/v1)
    MODEL_NAME     — model identifier (default: gpt-4o)
    HF_TOKEN       — HuggingFace / API key for the LLM

Usage:
    # Run all difficulty tiers (2 episodes each) — default for baseline scoring
    python inference.py --space https://Maverick98-sre-incident-env.hf.space

    # Run locally
    python inference.py

    # Single difficulty
    python inference.py --space https://Maverick98-sre-incident-env.hf.space --difficulty hard --episodes 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv(override=False)

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
MODEL = os.getenv("MODEL_NAME") or "gpt-4o"
MAX_STEPS = 200
CONTEXT_CHAR_LIMIT = 120000
VERBOSE = True

SYSTEM_PROMPT = """You are an expert on-call Site Reliability Engineer responding to a production incident.

# MISSION
Investigate the incident, identify the root cause, FIX the problem, and verify resolution.
This is a LIVE system — your actions have real consequences. Wrong fixes can make things worse.

# TOOLS

## Discovery (understand the system)
- list_services: See all services in the incident topology
- read_logs: Read logs for a service (filter by level: ERROR, WARN, INFO)
- check_metric: Check a metric time-series for a service
- get_service_info: Get the service runbook — available actions, config params, recent deploys

## Platform Remediation (standard SRE actions)
- restart_service: Bounce a service process (clears runtime state)
- rollback_deploy: Revert a service to its previous deployment version
- scale_replicas: Scale a service horizontally (add/remove instances)

## Application Remediation (service-specific actions)
- execute_runbook: Run a service-specific maintenance action discovered via get_service_info

## Resolution
- verify_resolution: Check if the system is healthy + submit your diagnosis

# PROTOCOL
1. ORIENT: list_services to see the topology.
2. INVESTIGATE: read_logs across services to find which are affected and trace the error chain upstream to the origin.
3. DISCOVER: get_service_info on suspected services to learn what actions, config params, and recent deploys are available.
4. REMEDIATE: Apply a fix based on your diagnosis. The right fix depends on what you found — there is no single approach that works everywhere.
5. OBSERVE: After ANY remediation, call read_logs to see what changed. The system will show you the outcome. Adjust your approach based on what you see.
6. VERIFY: When the system is healthy, call verify_resolution with your diagnosis.

# CRITICAL RULES
- DISCOVER BEFORE FIXING: Call get_service_info before using execute_runbook — it tells you valid action names and config keys.
- OBSERVE AFTER FIXING: Always read_logs after a remediation to check the outcome.
- CAUSE ≠ EFFECT: The service with the most errors is usually a VICTIM, not the cause. Trace upstream.
- WRONG FIXES HAVE CONSEQUENCES: The system will tell you if your action made things worse. Read the outcome and adjust.
"""


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


def mcp_tools_to_openai(tools) -> List[dict]:
    """Convert tool list (dicts or objects) to OpenAI function-calling format."""
    openai_tools = []
    for tool in tools:
        # Handle both dict (from HTTP) and object (from WebSocket) formats
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description", "")
            schema = tool.get("inputSchema", tool.get("input_schema", {}))
        else:
            name = tool.name
            description = tool.description or ""
            schema = tool.input_schema if hasattr(tool, "input_schema") else {}

        properties = {}
        required = []
        if schema and "properties" in schema:
            for pname, pschema in schema["properties"].items():
                prop = {"type": pschema.get("type", "string")}
                if "description" in pschema:
                    prop["description"] = pschema["description"]
                properties[pname] = prop
            required = schema.get("required", [])

        openai_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return openai_tools


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------


def summarize_old_messages(messages: List[dict]) -> List[dict]:
    """Replace old tool_call/tool_response pairs with a compact text summary."""
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    total_chars += sum(
        len(tc.get("function", {}).get("arguments", ""))
        for m in messages
        for tc in (m.get("tool_calls") or [])
    )
    if total_chars <= CONTEXT_CHAR_LIMIT:
        return messages

    system_msg = messages[0]
    initial_user_msg = messages[1]

    keep_recent = 15
    split_idx = len(messages) - keep_recent
    while split_idx < len(messages) - 4:
        if messages[split_idx].get("role") == "assistant":
            break
        split_idx += 1
    old_messages = messages[2:split_idx]
    recent_messages = messages[split_idx:]

    summary_lines = ["Previous investigation steps:"]
    i = 0
    while i < len(old_messages):
        msg = old_messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
                args_short = json.dumps(tool_args)[:80]
            except (json.JSONDecodeError, TypeError):
                args_short = tc["function"]["arguments"][:80]

            result_short = "(no response)"
            if i + 1 < len(old_messages) and old_messages[i + 1].get("role") == "tool":
                content = old_messages[i + 1].get("content", "")
                result_short = _summarize_tool_result(content)
                i += 1

            summary_lines.append(f"- {tool_name}({args_short}) → {result_short}")
        i += 1

    summary_text = "\n".join(summary_lines)

    return [
        system_msg,
        initial_user_msg,
        {"role": "user", "content": summary_text},
    ] + recent_messages


def _summarize_tool_result(content: str, max_chars: int = 150) -> str:
    """Summarize a tool result into a short description."""
    if not content or len(content) <= max_chars:
        return content or "(empty)"
    try:
        data = json.loads(content)
        if "logs" in data:
            return f"{data.get('count', len(data.get('logs', [])))} log entries"
        if "metric_series" in data:
            return f"{data.get('service', '?')}.{data.get('metric', '?')}: {data.get('points', '?')} points"
        if "services" in data:
            return f"services: {', '.join(data['services'][:6])}"
        if "error" in data:
            return f"error: {data['error'][:100]}"
        return json.dumps(data)[:max_chars]
    except (json.JSONDecodeError, TypeError):
        return content[:max_chars] + "..."


# ---------------------------------------------------------------------------
# Episode runner — works with both HTTP and WebSocket clients
# ---------------------------------------------------------------------------


async def run_episode(
    env,
    llm_client: OpenAI,
    model: str,
    tools: List[dict],
    difficulty: str = "medium",
    scenario_id: str = None,
    seed: int = None,
) -> Dict[str, Any]:
    """Run a single investigation episode using native function calling."""
    tool_names = [t["function"]["name"] for t in tools]
    is_reasoning = any(x in model for x in ["o3", "o4", "gpt-5"])

    # Reset episode
    reset_kwargs = {"difficulty": difficulty}
    if scenario_id:
        reset_kwargs["scenario_id"] = scenario_id
    if seed is not None:
        reset_kwargs["seed"] = seed
    reset_result = await env.reset(**reset_kwargs)

    # Hackathon Phase 2 structured output: [START]
    task_name = scenario_id or f"{difficulty}_episode"
    print(f"[START] task={task_name}", flush=True)

    # Extract alert message — works for both HTTP (dict) and WebSocket (Observation)
    if isinstance(reset_result, dict):
        alert_msg = reset_result.get("message", "")
    elif hasattr(reset_result, "metadata") and reset_result.metadata:
        alert_msg = reset_result.metadata.get("message", "")
    else:
        alert_msg = ""
    if not alert_msg:
        alert_msg = "Production incident detected. Use list_services to begin."

    chat_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": alert_msg},
    ]

    step_count = 0
    done = False
    consecutive_text = 0

    while not done and step_count < MAX_STEPS:
        step_count += 1

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": chat_history,
            "tools": tools,
            "tool_choice": "auto",
        }
        if is_reasoning:
            create_kwargs["max_completion_tokens"] = 2000
        else:
            create_kwargs["temperature"] = 0.1
            create_kwargs["max_tokens"] = 500

        try:
            response = llm_client.chat.completions.create(**create_kwargs)
        except Exception as e:
            if VERBOSE:
                print(f"    API error: {str(e)[:100]}")
            print(f"[END] task={task_name} score=0.0001 steps={step_count}", flush=True)
            return {"reward": 0.0001, "error": str(e)[:200], "steps": step_count}

        message = response.choices[0].message

        # Handle function call
        if message.tool_calls:
            consecutive_text = 0
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                chat_history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": tool_call_id, "type": "function",
                        "function": {"name": tool_name, "arguments": tool_call.function.arguments}}],
                })
                chat_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Error: malformed JSON arguments. Please retry with valid JSON.",
                })
                if VERBOSE:
                    print(f"    T{step_count}: {tool_name}(...) [bad JSON]")
                continue
        elif message.content:
            consecutive_text += 1
            if VERBOSE:
                print(f"    [text] {message.content[:80]}")
            chat_history.append({"role": "assistant", "content": message.content})
            if consecutive_text >= 3:
                chat_history.append({
                    "role": "user",
                    "content": "You MUST call verify_resolution NOW with your best diagnosis.",
                })
            else:
                chat_history.append({
                    "role": "user",
                    "content": "Please use one of the available tools.",
                })
            continue
        else:
            continue

        # Add tool call to history
        chat_history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": tool_call.function.arguments,
                },
            }],
        })

        if VERBOSE:
            args_short = json.dumps(tool_args)[:60]
            print(f"    T{step_count}: {tool_name}({args_short})", end="", flush=True)

        # Validate tool name
        if tool_name not in tool_names:
            chat_history.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": f"Unknown tool '{tool_name}'. Available: {tool_names}",
            })
            if VERBOSE:
                print(" [unknown tool]")
            continue

        # Execute tool — works for both HTTP and WebSocket clients
        try:
            result_text = await env.call_tool(tool_name, **tool_args)
        except Exception as e:
            chat_history.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": f"Error: {e}",
            })
            if VERBOSE:
                print(f" [error: {e}]")
            continue

        # Ensure result_text is a string
        if not isinstance(result_text, str):
            result_text = json.dumps(result_text) if result_text else ""

        # Check done/reward from result
        reward = 0.0
        if result_text:
            try:
                parsed = json.loads(result_text)
                if parsed.get("done"):
                    done = True
                if "reward" in parsed:
                    reward = float(parsed["reward"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Also check HTTP client's stored state
        if hasattr(env, "_last_done") and env._last_done:
            done = True
        if hasattr(env, "_last_reward") and env._last_reward:
            reward = max(reward, env._last_reward)

        if VERBOSE:
            if done:
                print(f" → done, reward={reward:.4f}")
            else:
                print()

        # Hackathon Phase 2 structured output: [STEP] — printed on its own line
        # Score must be strictly in (0, 1)
        safe_step_reward = max(0.0001, min(0.9999, reward))
        print(f"[STEP] step={step_count} reward={safe_step_reward:.4f}", flush=True)

        if done:
            safe_reward = max(0.0001, min(0.9999, reward))
            print(f"[END] task={task_name} score={safe_reward:.4f} steps={step_count}", flush=True)
            return {"reward": float(reward), "steps": step_count}

        # Truncate large results
        if result_text and len(result_text) > 3000:
            result_text = result_text[:3000] + "\n...(truncated)"

        chat_history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_text or "No result",
        })

        chat_history = summarize_old_messages(chat_history)

    print(f"[END] task={task_name} score=0.0001 steps={MAX_STEPS}", flush=True)
    return {"reward": 0.0001, "error": "max_turns", "steps": MAX_STEPS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="SRE Incident Env Inference")
    parser.add_argument("--difficulty", default=None,
                        choices=["easy", "medium", "hard", "expert"])
    parser.add_argument("--episodes", type=int, default=1,
                        help="Runs per scenario (default 1 — keeps validator under 20min budget)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--space", default=None, help="HF Space URL")
    parser.add_argument("--websocket", action="store_true",
                        help="Use WebSocket transport instead of HTTP")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: Set HF_TOKEN, OPENAI_API_KEY, or API_KEY.")
        sys.exit(1)

    model = args.model or MODEL
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Choose transport
    if args.space:
        base_url = args.space
        mode = f"remote ({args.space})"
    else:
        base_url = "http://localhost:8000"
        mode = "local (http://127.0.0.1:8000)"

    if args.websocket:
        from client import SREIncidentEnv
        env = SREIncidentEnv(base_url=base_url)
        mode += " [WebSocket]"
    else:
        from client import SREIncidentEnvHTTP
        env = SREIncidentEnvHTTP(base_url=base_url)
        mode += " [HTTP]"

    # Start local server if not using Space
    server_proc = None
    if not args.space:
        import subprocess, time
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app",
             "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(3)

    difficulties = [args.difficulty] if args.difficulty else ["easy", "medium", "hard"]

    BASELINE_SCENARIOS = {
        "easy": [
            "jvm_metaspace_classloader_leak_001",
            "etcd_compaction_quota_alarm_001",
        ],
        "medium": [
            "kafka_partition_rebalance_storm_001",
            "cpu_microcode_tsc_drift_001",
            "cert_expiry_mutual_tls_001",
        ],
        "hard": [
            "numa_cross_socket_latency_001",
            "kernel_tcp_rmem_silent_drop_001",
            "wal_archive_disk_full_h002",
        ],
    }

    try:
        # Discover tools using a one-shot session (separate from episodes)
        async with env as discover_env:
            reset_result = await discover_env.reset(difficulty="easy")
            tools_raw = await discover_env.list_tools()
            tools = mcp_tools_to_openai(tools_raw)

        if VERBOSE:
            print(f"Mode: {mode}")
            print(f"Model: {model}")
            print(f"Tools: {[t['function']['name'] for t in tools]}")
            print(f"Difficulties: {difficulties} | Episodes per tier: {args.episodes}")
            print("=" * 60)

        all_results: Dict[str, List[Dict[str, Any]]] = {}

        # Helper to build a fresh client per episode (HTTP transport — each episode gets its own session)
        def make_env():
            if args.websocket:
                from client import SREIncidentEnv
                return SREIncidentEnv(base_url=base_url)
            else:
                from client import SREIncidentEnvHTTP
                return SREIncidentEnvHTTP(base_url=base_url)

        for difficulty in difficulties:
            print(f"\n{'─' * 40}")
            print(f"  Difficulty: {difficulty.upper()}")
            print(f"{'─' * 40}")

            tier_results = []
            scenario_ids = BASELINE_SCENARIOS.get(difficulty, [None])
            # Run every scenario in the tier, args.episodes times each
            total = len(scenario_ids) * args.episodes
            idx = 0
            for sid in scenario_ids:
                for run_num in range(1, args.episodes + 1):
                    idx += 1
                    print(f"\n  Episode {idx}/{total} ({sid} run{run_num}):")
                    try:
                        async with make_env() as ep_env:
                            result = await run_episode(
                                ep_env, llm_client, model, tools, difficulty, scenario_id=sid,
                            )
                    except Exception as e:
                        if VERBOSE:
                            print(f"    SESSION FAILED: {str(e)[:120]}")
                        task_name = sid or f"{difficulty}_episode"
                        print(f"[START] task={task_name}", flush=True)
                        print(f"[END] task={task_name} score=0.0001 steps=0", flush=True)
                        result = {"reward": 0.0001, "error": f"session_error: {str(e)[:100]}", "steps": 0}
                    result["scenario_id"] = sid
                    result["run"] = run_num
                    tier_results.append(result)
            all_results[difficulty] = tier_results

            # Summary
            print(f"\n{'=' * 60}")
            print(f"BASELINE RESULTS — {model}")
            print(f"{'=' * 60}")
            overall_rewards = []
            for difficulty, results in all_results.items():
                valid = [r for r in results if "error" not in r]
                avg = sum(r["reward"] for r in valid) / len(valid) if valid else 0
                errors = len(results) - len(valid)
                overall_rewards.extend(r["reward"] for r in valid)
                print(f"  {difficulty:8s}: avg={avg:.4f} ({len(valid)} valid, {errors} errors)")
                for i, r in enumerate(results):
                    status = f"reward={r['reward']:.4f}" if "error" not in r else f"error={r['error'][:40]}"
                    print(f"    Episode {i+1}: {status}")

            if overall_rewards:
                print(f"\n  OVERALL: avg={sum(overall_rewards)/len(overall_rewards):.4f} across {len(overall_rewards)} episodes")

    finally:
        if server_proc:
            server_proc.terminate()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
