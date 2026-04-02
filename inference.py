#!/usr/bin/env python3
"""Baseline inference for SRE Incident Response environment.

Connects to the environment via HF Space URL (remote) or runs locally.
Uses native OpenAI function calling for tool use.

Required env vars:
    API_BASE_URL   — LLM endpoint (default: https://api.openai.com/v1)
    MODEL_NAME     — model identifier (default: gpt-4o)
    HF_TOKEN       — HuggingFace / API key for the LLM

Usage:
    # Against remote HF Space (how judges run it):
    python inference.py --space https://Maverick98-sre-incident-env.hf.space

    # Local (faster, for development):
    python inference.py

    # Options:
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
# For LLM calls: prefer explicit API keys over HF_TOKEN
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL = os.getenv("MODEL_NAME") or "gpt-4o"
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
- RED HERRINGS exist: some services show coincidental degradation from background jobs
  or unrelated deploys. If issues don't connect to the main causal chain, ignore them.
- Budget your queries: ~40% scanning logs, ~40% checking metrics, save 1 for diagnosis.
- NEVER repeat a tool call with identical arguments.
- Check metrics on the suspected root cause, not just on the symptomatic services.

# DIAGNOSIS FORMAT
When calling submit_diagnosis, provide ALL fields:
- affected_service: The service where the root defect ORIGINATES (not the loudest symptom).
- failure_type: Category from this taxonomy: oom_kill, connection_pool, connection_leak, config_drift, slow_external_api, gc_pressure, disk_full, n_plus_one_query, thread_pool_starvation, cert_expiry, cache_stampede, cache_node_failure, replication_lag, rate_limit_breach, dns_misconfiguration, thundering_herd_deploy, clock_skew_jwt, library_version_conflict, split_brain_db, circular_dependency_deadlock, bad_index_drop
- root_cause: "[service] [mechanism] caused [downstream effect chain]"
- causal_chain: Comma-separated service chain from root cause to visible symptom, e.g. "product-cache,checkout-service,api-gateway"
- confidence: 0.0 to 1.0

GOOD diagnosis:
  affected_service: "worker-pool"
  failure_type: "oom_kill"
  root_cause: "worker-pool memory leak in order deserialization caused repeated OOM kills"
  causal_chain: "worker-pool,order-service,api-gateway"

BAD: affected_service="api-gateway", failure_type="other" (symptom service, wrong type)
"""


# ---------------------------------------------------------------------------
# Tool conversion (MCP tools → OpenAI function calling format)
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
                if "default" in schema:
                    prop["default"] = schema["default"]
                if "enum" in schema:
                    prop["enum"] = schema["enum"]
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
# Episode runner (works with both local and remote env)
# ---------------------------------------------------------------------------


async def run_episode(
    env,
    llm_client: OpenAI,
    model: str,
    tools: List[dict],
    difficulty: str = "medium",
) -> Dict[str, Any]:
    """Run a single investigation episode against any OpenEnv MCP environment."""
    from openenv.core.env_server.mcp_types import CallToolAction

    tool_names = [t["function"]["name"] for t in tools]
    is_reasoning = any(x in model for x in ["o3", "o4", "gpt-5"])

    # Reset episode
    result = await env.reset(difficulty=difficulty)
    # Extract alert message — handle both StepResult and Observation formats
    alert_msg = ""
    if hasattr(result, "observation"):
        obs_obj = result.observation
        if hasattr(obs_obj, "metadata") and obs_obj.metadata:
            alert_msg = obs_obj.metadata.get("message", "")
    if hasattr(result, "metadata") and result.metadata:
        alert_msg = alert_msg or result.metadata.get("message", "")
    if not alert_msg:
        alert_msg = "Production incident detected. You have a limited query budget. Use list_services to begin investigation."

    chat_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": alert_msg},
    ]

    step_count = 0
    max_steps = 20
    done = False
    consecutive_text = 0  # Track text-only responses to prevent loops

    while not done and step_count < max_steps:
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
            consecutive_text = 0  # Reset on successful tool call
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
                # Model is stuck — force it to submit or give up
                chat_history.append({
                    "role": "user",
                    "content": "You must submit your diagnosis NOW using submit_diagnosis. "
                               "If you cannot determine the root cause, submit your best guess.",
                })
            else:
                chat_history.append({
                    "role": "user",
                    "content": "Please use one of the available tools to continue investigating.",
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

        # Execute in environment (async)
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

        # Extract from StepResult → observation → result
        obs = step_result.observation if hasattr(step_result, "observation") else step_result

        # Extract tool result text from various formats
        result_text = ""
        # CallToolObservation has .result dict with content/data
        obs_result = getattr(obs, "result", None)
        if obs_result:
            if isinstance(obs_result, dict):
                # Remote format: {"content": [{"text": "..."}], "data": "..."}
                if "data" in obs_result:
                    result_text = obs_result["data"]
                elif "content" in obs_result:
                    for item in obs_result["content"]:
                        if isinstance(item, dict) and "text" in item:
                            result_text = item["text"]
                            break
            elif hasattr(obs_result, "data"):
                # Local format: CallToolResult object
                result_text = obs_result.data
        if not result_text and hasattr(obs, "metadata") and obs.metadata:
            result_text = json.dumps(obs.metadata)

        # Parse done/reward from tool result text (works for both local and remote)
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
            return {
                "reward": float(reward),
                "steps": step_count,
            }

        # Feed result back
        chat_history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_text or "No result",
        })

        # Keep history compact
        if len(chat_history) > 30:
            chat_history = [chat_history[0]] + chat_history[-29:]

    return {"reward": 0.0, "error": "max_turns", "steps": step_count}


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
                        help="HF Space URL (e.g. https://Maverick98-sre-incident-env.hf.space). "
                             "If omitted, runs environment locally.")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: Set HF_TOKEN, API_KEY, or OPENAI_API_KEY.")
        sys.exit(1)

    model = args.model or MODEL
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to environment: remote HF Space or local
    if args.space:
        from client import SREIncidentEnv
        env = SREIncidentEnv(base_url=args.space)
        mode = f"remote ({args.space})"
    else:
        # Local: use the env directly but wrap with async client interface
        from client import SREIncidentEnv
        env = SREIncidentEnv(base_url="http://localhost:8000")
        # Start local server in background
        import subprocess
        import time
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app",
             "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(3)  # Wait for server startup
        mode = "local (http://127.0.0.1:8000)"

    try:
        # Discover tools
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

        # Summary
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
        if not args.space and 'server_proc' in dir():
            server_proc.terminate()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
