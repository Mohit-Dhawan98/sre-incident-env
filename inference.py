#!/usr/bin/env python3
"""Baseline inference for SRE Incident Response environment.

Uses finqa pattern (native function calling with message chaining).
Adds smart summarization when context exceeds threshold — replaces old
tool_call/tool_response pairs with a compact investigation summary.

Required env vars:
    API_BASE_URL   — LLM endpoint (default: https://api.openai.com/v1)
    MODEL_NAME     — model identifier (default: gpt-4o)
    HF_TOKEN       — HuggingFace / API key for the LLM

Usage:
    python inference.py
    python inference.py --space https://Maverick98-sre-incident-env.hf.space
    python inference.py --difficulty hard --episodes 3 --model o4-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL = os.getenv("MODEL_NAME") or "gpt-4o"
MAX_STEPS = 20
CONTEXT_CHAR_LIMIT = 120000  # ~30k tokens — summarize when total chars exceed this
VERBOSE = True

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer investigating a production incident.

# MISSION
Identify the ROOT CAUSE service and the specific mechanism that triggered this incident.
The root cause is NOT the service showing the most visible errors — it is the upstream
service whose failure CAUSED those errors. Trace the causal chain to its origin.

# INVESTIGATION PROTOCOL
Phase 1 - ORIENT (1 call): Call list_services to see the service topology.
Phase 2 - SCAN (2-3 calls): Read ERROR logs for 2-3 services. Identify WHICH services
  show errors and WHAT upstream services those errors reference.
Phase 3 - TRACE (2-3 calls): For each error chain, check metrics on the SUSPECTED root
  cause service. Look for patterns:
  - Ramp = resource leak or gradual exhaustion
  - Step change = config change or deployment
  - Spike = traffic burst or retry storm
Phase 4 - DIAGNOSE (1 call): Submit only when you can answer ALL THREE:
  1. WHICH service has the original defect?
  2. WHAT specific mechanism failed? (not just "errors occurred")
  3. WHAT downstream chain did it cause?

# CRITICAL RULES
- CAUSE ≠ EFFECT: If service-A logs errors mentioning service-B, the problem likely
  originates in service-B. Investigate service-B, do not blame service-A.
- The service with the MOST VISIBLE errors is usually a VICTIM, not the cause.
- RED HERRINGS exist: some services show coincidental degradation. Ignore if unconnected.
- Budget your queries: ~40% scanning logs, ~40% checking metrics, save 1 for diagnosis.
- NEVER repeat a tool call with identical arguments.

# DIAGNOSIS FORMAT
When calling submit_diagnosis, provide ALL fields:
- affected_service: The service where the root defect ORIGINATES (not the loudest symptom).
- failure_type: Category from: oom_kill, connection_pool, connection_leak, config_drift, slow_external_api, gc_pressure, disk_full, n_plus_one_query, thread_pool_starvation, cert_expiry, cache_stampede, cache_node_failure, replication_lag, rate_limit_breach, dns_misconfiguration, thundering_herd_deploy, clock_skew_jwt, library_version_conflict, split_brain_db, circular_dependency_deadlock, bad_index_drop
- root_cause: "[service] [mechanism] caused [downstream effect chain]"
- causal_chain: Comma-separated service chain from root to visible symptom
- confidence: 0.0 to 1.0
"""


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


def mcp_tools_to_openai(tools) -> List[dict]:
    """Convert MCP tool list to OpenAI function-calling format."""
    openai_tools = []
    for tool in tools:
        properties = {}
        required = []
        if tool.input_schema and "properties" in tool.input_schema:
            for name, schema in tool.input_schema["properties"].items():
                prop = {"type": schema.get("type", "string")}
                if "description" in schema:
                    prop["description"] = schema["description"]
                properties[name] = prop
            required = tool.input_schema.get("required", [])
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return openai_tools


# ---------------------------------------------------------------------------
# Context management — summarize old history, keep recent pairs intact
# ---------------------------------------------------------------------------


def summarize_old_messages(messages: List[dict]) -> List[dict]:
    """Replace old tool_call/tool_response pairs with a compact text summary.

    Keeps: system prompt + initial user message + summary + recent messages.
    This preserves the finqa function-calling pattern for recent interactions
    while compressing old context.
    """
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

    # Split: old messages to summarize vs recent messages to keep
    # Find a safe split point — must start recent window at an assistant message
    # (never at a tool response, which would be orphaned)
    keep_recent = 15
    split_idx = len(messages) - keep_recent
    # Walk forward to find an assistant message (start of a pair)
    while split_idx < len(messages) - 4:
        if messages[split_idx].get("role") == "assistant":
            break
        split_idx += 1
    old_messages = messages[2:split_idx]
    recent_messages = messages[split_idx:]

    # Build summary from old tool interactions
    summary_lines = ["Previous investigation steps:"]
    i = 0
    while i < len(old_messages):
        msg = old_messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Extract tool call info
            tc = msg["tool_calls"][0]
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
                args_short = json.dumps(tool_args)[:80]
            except (json.JSONDecodeError, TypeError):
                args_short = tc["function"]["arguments"][:80]

            # Get the tool response (next message)
            result_short = "(no response)"
            if i + 1 < len(old_messages) and old_messages[i + 1].get("role") == "tool":
                content = old_messages[i + 1].get("content", "")
                result_short = _summarize_tool_result(content)
                i += 1  # skip the tool response

            summary_lines.append(f"- {tool_name}({args_short}) → {result_short}")
        elif msg.get("role") == "user":
            # Skip user nudge messages
            pass
        elif msg.get("role") == "assistant" and msg.get("content"):
            # Text response from model — skip in summary
            pass
        i += 1

    summary_text = "\n".join(summary_lines)

    # Rebuild: system + initial + summary + recent (intact function calling pairs)
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
            count = data.get("count", len(data.get("logs", [])))
            return f"{count} log entries"
        if "metric_series" in data:
            pts = data.get("points", len(data.get("metric_series", [])))
            svc = data.get("service", "?")
            metric = data.get("metric", "?")
            return f"{svc}.{metric}: {pts} points"
        if "services" in data:
            return f"services: {', '.join(data['services'][:6])}"
        if "error" in data:
            return f"error: {data['error'][:100]}"
        if "result" in data:
            return f"result: {data['result'][:100]}"
        return json.dumps(data)[:max_chars]
    except (json.JSONDecodeError, TypeError):
        return content[:max_chars] + "..."


# ---------------------------------------------------------------------------
# Episode runner (finqa pattern + smart summarization)
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
    from openenv.core.env_server.mcp_types import CallToolAction

    tool_names = [t["function"]["name"] for t in tools]
    is_reasoning = any(x in model for x in ["o3", "o4", "gpt-5"])

    # Reset episode
    reset_kwargs = {"difficulty": difficulty}
    if scenario_id:
        reset_kwargs["scenario_id"] = scenario_id
    if seed is not None:
        reset_kwargs["seed"] = seed
    result = await env.reset(**reset_kwargs)

    # Extract alert message
    alert_msg = ""
    if hasattr(result, "observation") and hasattr(result.observation, "metadata"):
        alert_msg = result.observation.metadata.get("message", "")
    if hasattr(result, "metadata") and result.metadata:
        alert_msg = alert_msg or result.metadata.get("message", "")
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

        # LLM call with function calling
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
            return {"reward": 0.0, "error": str(e)[:200], "steps": step_count}

        message = response.choices[0].message

        # Handle function call
        if message.tool_calls:
            consecutive_text = 0
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_call_id = tool_call.id
        elif message.content:
            consecutive_text += 1
            if VERBOSE:
                print(f"    [text] {message.content[:80]}")
            chat_history.append({"role": "assistant", "content": message.content})
            if consecutive_text >= 3:
                chat_history.append({
                    "role": "user",
                    "content": "You MUST call submit_diagnosis NOW with your best guess.",
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

        # Execute in environment
        try:
            action = CallToolAction(tool_name=tool_name, arguments=tool_args)
            step_result = await env.step(action)
        except Exception as e:
            chat_history.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": f"Error: {e}",
            })
            if VERBOSE:
                print(f" [error: {e}]")
            continue

        # Extract result
        obs = step_result.observation if hasattr(step_result, "observation") else step_result
        result_text = ""
        obs_result = getattr(obs, "result", None)
        if obs_result:
            if isinstance(obs_result, dict) and "data" in obs_result:
                result_text = obs_result["data"]
            elif hasattr(obs_result, "data"):
                result_text = obs_result.data
        if not result_text and hasattr(obs, "metadata") and obs.metadata:
            result_text = json.dumps(obs.metadata)

        # Check done/reward from tool result
        done = getattr(step_result, "done", False) or getattr(obs, "done", False)
        reward = getattr(step_result, "reward", 0.0) or getattr(obs, "reward", 0.0)
        if result_text:
            try:
                parsed = json.loads(result_text)
                if parsed.get("done"):
                    done = True
                if "reward" in parsed:
                    reward = float(parsed["reward"])
            except (json.JSONDecodeError, TypeError):
                pass

        if VERBOSE:
            if done:
                print(f" → done, reward={reward:.4f}")
            else:
                print()

        if done:
            return {"reward": float(reward), "steps": step_count}

        # Truncate large tool results before adding to history
        if result_text and len(result_text) > 3000:
            result_text = result_text[:3000] + "\n...(truncated)"

        # Feed result back
        chat_history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_text or "No result",
        })

        # Summarize old context AFTER tool response is added (pairs intact)
        chat_history = summarize_old_messages(chat_history)

    return {"reward": 0.0, "error": "max_turns", "steps": MAX_STEPS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="SRE Incident Env Inference")
    parser.add_argument("--difficulty", default="medium",
                        choices=["easy", "medium", "hard", "expert"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--space", default=None,
                        help="HF Space URL. If omitted, runs locally.")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: Set OPENAI_API_KEY, API_KEY, or HF_TOKEN.")
        sys.exit(1)

    model = args.model or MODEL
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to environment
    if args.space:
        from client import SREIncidentEnv
        env = SREIncidentEnv(base_url=args.space)
        mode = f"remote ({args.space})"
    else:
        from client import SREIncidentEnv
        env = SREIncidentEnv(base_url="http://localhost:8000")
        import subprocess, time
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app",
             "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
        mode = "local (http://127.0.0.1:8000)"

    try:
        await env.reset(difficulty=args.difficulty)
        mcp_tools = await env.list_tools()
        tools = mcp_tools_to_openai(mcp_tools)

        if VERBOSE:
            print(f"Mode: {mode}")
            print(f"Model: {model}")
            print(f"Tools: {[t['function']['name'] for t in tools]}")
            print(f"Difficulty: {args.difficulty} | Episodes: {args.episodes}")
            print("=" * 60)

        results = []
        for i in range(args.episodes):
            print(f"\nEpisode {i+1}/{args.episodes}:")
            result = await run_episode(env, llm_client, model, tools, args.difficulty)
            results.append(result)

        valid = [r for r in results if "error" not in r]
        avg = sum(r["reward"] for r in valid) / len(valid) if valid else 0
        errors = len(results) - len(valid)

        print(f"\n{'=' * 60}")
        print(f"Results ({args.difficulty}, {model}):")
        for i, r in enumerate(results):
            status = f"reward={r['reward']:.4f}" if "error" not in r else f"error={r['error'][:40]}"
            print(f"  Episode {i+1}: {status}")
        print(f"  Average: {avg:.4f} ({len(valid)} valid, {errors} errors)")

    finally:
        await env.close()
        if not args.space and 'server_proc' in locals():
            server_proc.terminate()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
