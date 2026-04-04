"""Detailed benchmark — captures full model reasoning + tool call/response pairs.

Stores per-scenario JSON with complete conversation history for behavior analysis.
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from client import SREIncidentEnv
from inference import SYSTEM_PROMPT, MAX_STEPS, mcp_tools_to_openai

VERBOSE = True


async def run_episode_detailed(
    env, llm_client, model, tools, difficulty, scenario_id, space_url,
):
    """Run episode capturing full conversation trace."""
    from openenv.core.env_server.mcp_types import CallToolAction
    from inference import summarize_old_messages

    CONTEXT_CHAR_LIMIT = 80_000
    tool_names = [t["function"]["name"] for t in tools]
    is_reasoning = any(x in model for x in ["o3", "o4", "gpt-5"])

    # Reset
    result = await env.reset(difficulty=difficulty, scenario_id=scenario_id)
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

    # Full trace: every step with model reasoning + tool response
    trace = []
    step_count = 0
    done = False
    consecutive_text = 0
    reward = 0.0

    while not done and step_count < MAX_STEPS:
        step_count += 1

        # Compress if needed
        chat_history = summarize_old_messages(chat_history)

        create_kwargs = {
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
            trace.append({"step": step_count, "type": "api_error", "error": str(e)[:500]})
            return {"reward": 0.0, "error": str(e)[:200], "steps": step_count, "trace": trace}

        message = response.choices[0].message

        # Capture model's text reasoning (if any)
        model_text = message.content or ""

        if message.tool_calls:
            consecutive_text = 0
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id

            try:
                tool_args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                trace.append({
                    "step": step_count,
                    "type": "bad_json",
                    "tool": tool_name,
                    "raw_args": tool_call.function.arguments[:500],
                    "model_reasoning": model_text,
                })
                chat_history.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": tool_call_id, "type": "function",
                        "function": {"name": tool_name, "arguments": tool_call.function.arguments}}],
                })
                chat_history.append({
                    "role": "tool", "tool_call_id": tool_call_id,
                    "content": f"Error: malformed JSON arguments. Please retry with valid JSON.",
                })
                continue

            # Add to chat history
            chat_history.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": tool_call_id, "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}],
            })

            if tool_name not in tool_names:
                chat_history.append({
                    "role": "tool", "tool_call_id": tool_call_id,
                    "content": f"Unknown tool '{tool_name}'. Available: {tool_names}",
                })
                trace.append({
                    "step": step_count, "type": "unknown_tool",
                    "tool": tool_name, "args": tool_args,
                    "model_reasoning": model_text,
                })
                continue

            # Execute
            try:
                action = CallToolAction(tool_name=tool_name, arguments=tool_args)
                step_result = await env.step(action)
            except Exception as e:
                chat_history.append({
                    "role": "tool", "tool_call_id": tool_call_id,
                    "content": f"Error: {e}",
                })
                trace.append({
                    "step": step_count, "type": "env_error",
                    "tool": tool_name, "args": tool_args, "error": str(e)[:300],
                    "model_reasoning": model_text,
                })
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

            done = getattr(step_result, "done", False) or getattr(obs, "done", False)
            if result_text:
                try:
                    parsed = json.loads(result_text)
                    if parsed.get("done"):
                        done = True
                    if "reward" in parsed:
                        reward = float(parsed["reward"])
                except (json.JSONDecodeError, TypeError):
                    pass

            # For non-huge results, store full response. For logs/metrics, truncate.
            result_for_trace = result_text
            if len(result_text) > 2000:
                try:
                    parsed = json.loads(result_text)
                    if "logs" in parsed:
                        log_count = parsed.get("count", len(parsed.get("logs", [])))
                        # Keep first 3 and last 3 log messages
                        logs = parsed.get("logs", [])
                        if len(logs) > 6:
                            sample = logs[:3] + [{"...": f"{len(logs)-6} more entries"}] + logs[-3:]
                        else:
                            sample = logs
                        result_for_trace = json.dumps({"logs_sample": sample, "count": log_count})
                    elif "metric_series" in parsed:
                        pts = parsed.get("points", 0)
                        series = parsed.get("metric_series", [])
                        sample = series[:3] + series[-3:] if len(series) > 6 else series
                        result_for_trace = json.dumps({
                            "metric": parsed.get("metric"), "service": parsed.get("service"),
                            "points": pts, "sample": sample,
                        })
                except (json.JSONDecodeError, TypeError):
                    result_for_trace = result_text[:2000] + "...[truncated]"

            trace.append({
                "step": step_count,
                "type": "tool_call",
                "tool": tool_name,
                "args": tool_args,
                "result": result_for_trace,
                "model_reasoning": model_text,
                "done": done,
            })

            chat_history.append({
                "role": "tool", "tool_call_id": tool_call_id, "content": result_text,
            })

            if VERBOSE:
                args_short = json.dumps(tool_args)[:70]
                done_str = f" → done, reward={reward:.4f}" if done else ""
                print(f"    T{step_count}: {tool_name}({args_short}){done_str}")

        elif model_text:
            consecutive_text += 1
            trace.append({
                "step": step_count,
                "type": "text_response",
                "model_reasoning": model_text,
            })
            chat_history.append({"role": "assistant", "content": model_text})

            if VERBOSE:
                print(f"    [text] {model_text[:100]}")

            if consecutive_text >= 3:
                chat_history.append({
                    "role": "user",
                    "content": "You MUST call verify_resolution NOW with your best diagnosis. If you haven't fixed the system yet, call it anyway — you'll get partial credit for a correct diagnosis.",
                })
            else:
                chat_history.append({
                    "role": "user",
                    "content": "Please use one of the available tools.",
                })
        else:
            trace.append({"step": step_count, "type": "empty_response"})

    return {
        "reward": reward,
        "steps": step_count,
        "done": done,
        "trace": trace,
    }


async def main():
    space = os.environ.get("SPACE", "http://localhost:8000")
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_tag = os.environ.get("MODEL_TAG", model.replace("/", "_"))

    llm = OpenAI(base_url=api_base, api_key=api_key)
    env = SREIncidentEnv(base_url=space)

    await env.reset(difficulty="easy")
    mcp_tools = await env.list_tools()
    tools = mcp_tools_to_openai(mcp_tools)

    # Load V2 scenarios
    v2_path = Path(__file__).parent / "scenarios" / "incidents_v2.jsonl"
    v1_path = Path(__file__).parent / "scenarios" / "incidents.jsonl"
    scenario_path = v2_path if v2_path.exists() else v1_path
    with open(scenario_path) as f:
        scenarios = [json.loads(l) for l in f]

    print(f"Model: {model}")
    print(f"Space: {space}")
    print(f"Scenarios: {len(scenarios)}")
    print("=" * 60)

    # Output dir
    out_dir = Path(__file__).parent / "outputs" / f"{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_by_tier = {}
    all_results = []

    for s in scenarios:
        sid = s["id"]
        diff = s["difficulty"]
        optimal = s.get("failure", {}).get("remediation", {}).get("optimal_steps", 0)
        print(f"\n  {sid} ({diff}, {optimal}-step):", flush=True)

        # Fresh connection per scenario — prevents WebSocket timeout carryover
        try:
            await env.close()
        except Exception:
            pass
        env = SREIncidentEnv(base_url=space)

        t0 = time.time()
        r = await run_episode_detailed(env, llm, model, tools, diff, sid, space)
        elapsed = time.time() - t0

        score = r.get("reward", 0.0)
        steps = r.get("steps", 0)
        err = r.get("error", "")

        # Analyze trace
        trace = r.get("trace", [])
        tool_calls = [t for t in trace if t["type"] == "tool_call"]
        remediation_tools = [t for t in tool_calls if t["tool"] in
                             ("restart_service", "rollback_deploy", "scale_replicas", "execute_runbook")]
        investigation_tools = [t for t in tool_calls if t["tool"] in
                               ("read_logs", "check_metric", "list_services", "get_service_info")]
        text_responses = [t for t in trace if t["type"] == "text_response"]

        # Check if system was fixed
        system_fixed = False
        final_tool = tool_calls[-1] if tool_calls else None
        if final_tool and final_tool.get("result"):
            try:
                parsed = json.loads(final_tool["result"])
                system_fixed = parsed.get("system_healthy", False)
            except (json.JSONDecodeError, TypeError):
                pass

        summary = {
            "scenario_id": sid,
            "difficulty": diff,
            "optimal_steps": optimal,
            "reward": score,
            "total_steps": steps,
            "elapsed_seconds": round(elapsed, 1),
            "investigation_calls": len(investigation_tools),
            "remediation_calls": len(remediation_tools),
            "text_responses": len(text_responses),
            "system_fixed": system_fixed,
            "error": err,
            "remediation_actions": [
                {"tool": t["tool"], "args": t["args"],
                 "result_summary": (t.get("result", ""))[:200]}
                for t in remediation_tools
            ],
        }

        # Save full trace per scenario
        scenario_output = {
            "summary": summary,
            "trace": trace,
        }
        scenario_file = out_dir / f"{sid}.json"
        with open(scenario_file, "w") as f:
            json.dump(scenario_output, f, indent=2)

        status = f"reward={score:.4f}" if not err else f"error={err[:40]}"
        fix_str = "FIXED" if system_fixed else "NOT FIXED"
        print(f"    {status} | {fix_str} | inv={len(investigation_tools)} rem={len(remediation_tools)} | {elapsed:.1f}s")

        if diff not in results_by_tier:
            results_by_tier[diff] = []
        results_by_tier[diff].append(summary)
        all_results.append(summary)

    # Save overall results
    print(f"\n{'=' * 60}")
    print(f"RESULTS — {model}")
    print(f"{'=' * 60}")

    tier_averages = {}
    for tier in ["easy", "medium", "hard", "expert"]:
        rs = results_by_tier.get(tier, [])
        valid = [r for r in rs if not r["error"]]
        avg = sum(r["reward"] for r in valid) / len(valid) if valid else 0
        fixed = sum(1 for r in valid if r["system_fixed"])
        tier_averages[tier] = {"avg_reward": round(avg, 4), "fixed": fixed, "total": len(valid)}
        print(f"  {tier:8s}: avg={avg:.4f} | fixed={fixed}/{len(valid)}")
        for r in rs:
            s = f"reward={r['reward']:.4f}" if not r["error"] else f"error={r['error'][:30]}"
            fix = "OK" if r["system_fixed"] else "MISS"
            print(f"    [{fix}] {r['scenario_id']}: {s} (inv={r['investigation_calls']} rem={r['remediation_calls']})")

    all_valid = [r for r in all_results if not r["error"]]
    overall = sum(r["reward"] for r in all_valid) / len(all_valid) if all_valid else 0
    total_fixed = sum(1 for r in all_valid if r["system_fixed"])
    print(f"\n  OVERALL: {overall:.4f} | fixed={total_fixed}/{len(all_valid)}")

    # Save combined results
    combined = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "tier_averages": tier_averages,
        "overall_avg": round(overall, 4),
        "total_fixed": total_fixed,
        "scenarios": all_results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n  Detailed traces saved to: {out_dir}")
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
