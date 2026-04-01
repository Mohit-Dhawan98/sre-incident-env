"""
Benchmark frontier LLMs on SRE Incident Response environment.
Tests multiple models across difficulties to measure learning signal.

Usage:
    python benchmark.py
"""

import json
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from server.environment import SREIncidentEnvironment
from openenv.core.env_server.mcp_types import CallToolAction


SYSTEM_PROMPT = """You are an expert Site Reliability Engineer investigating a production incident.

You have access to these tools (call them by returning JSON):
- list_services: See available services (free, no query cost)
- read_logs: Read logs for a service. Args: service (str), window_minutes (int, default 5), level_filter (str or null)
- check_metric: Check a metric. Args: service (str), metric (str), window_minutes (int, default 10)
- submit_diagnosis: Submit your diagnosis. Args: root_cause (str), affected_service (str), confidence (float 0-1)

Investigation strategy:
1. Start with list_services to see the topology
2. Read ERROR and WARN logs to identify symptomatic services
3. Check metrics for anomalies — look for spikes, ramps, step changes
4. Trace the causal chain: visible symptoms → intermediate effects → root cause
5. The root cause is often 2-3 hops from the most visible symptom
6. Watch for red herrings: coincidental degradation in unrelated services
7. Submit a precise diagnosis: WHO did WHAT causing WHAT

IMPORTANT: Respond with ONLY a JSON object. No markdown, no explanation, no text before or after.
Examples:
{"tool": "list_services"}
{"tool": "read_logs", "args": {"service": "api-gateway", "window_minutes": 15, "level_filter": "ERROR"}}
{"tool": "submit_diagnosis", "args": {"root_cause": "db connection pool exhausted due to leak in payment-service v2.3.1", "affected_service": "db-primary", "confidence": 0.85}}
"""


def parse_llm_action(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise


def format_tool_result(obs) -> str:
    if hasattr(obs, 'result') and obs.result and hasattr(obs.result, 'data'):
        return obs.result.data
    if hasattr(obs, 'metadata') and obs.metadata:
        return json.dumps(obs.metadata, indent=2)
    return str(obs)


def run_episode(env, llm, model_name: str, scenario_id: str, difficulty: str, verbose: bool = True):
    """Run one episode. Returns dict with reward, steps, diagnosis, etc."""
    obs = env.reset(seed=42, difficulty=difficulty, scenario_id=scenario_id)
    alert_msg = obs.metadata.get("message", "Incident detected.")
    true_service = env._scenario["failure"]["root_service"]
    true_cause = env._scenario["failure"]["root_cause_statement"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": alert_msg},
    ]

    tool_calls = []
    parse_errors = 0
    start_time = time.time()

    for turn in range(20):
        try:
            # Reasoning models (o3, o1) use max_completion_tokens, no temperature
            is_reasoning = any(x in model_name for x in ["o3", "o1"])
            create_kwargs = {
                "model": model_name,
                "messages": messages,
            }
            if is_reasoning:
                create_kwargs["max_completion_tokens"] = 1000
            else:
                create_kwargs["temperature"] = 0.1
                create_kwargs["max_tokens"] = 500
            response = llm.chat.completions.create(**create_kwargs)
        except Exception as e:
            if verbose:
                print(f"    API error: {e}")
            return {"reward": 0.0, "error": str(e), "steps": turn}

        assistant_text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_text})

        try:
            action = parse_llm_action(assistant_text)
        except (json.JSONDecodeError, ValueError):
            parse_errors += 1
            if parse_errors >= 3:
                if verbose:
                    print(f"    Too many parse errors, aborting")
                return {"reward": 0.0, "error": "parse_failures", "steps": turn}
            messages.append({
                "role": "user",
                "content": "Invalid JSON. Respond with ONLY a JSON object like {\"tool\": \"list_services\"}",
            })
            continue

        tool_name = action.get("tool", "")
        tool_args = action.get("args", {})
        tool_calls.append({"tool": tool_name, "args": tool_args})

        if verbose:
            args_short = json.dumps(tool_args)[:50]
            print(f"    T{turn+1}: {tool_name}({args_short})")

        try:
            obs = env.step(CallToolAction(tool_name=tool_name, arguments=tool_args))
        except Exception as e:
            if verbose:
                print(f"    Step error: {e}")
            messages.append({"role": "user", "content": f"Error: {e}. Try a different action."})
            continue

        if obs.done:
            elapsed = time.time() - start_time
            # Extract submitted diagnosis from last tool call
            submitted_service = tool_args.get("affected_service", "") if tool_name == "submit_diagnosis" else ""
            submitted_cause = tool_args.get("root_cause", "") if tool_name == "submit_diagnosis" else ""

            result = {
                "reward": float(obs.reward),
                "steps": turn + 1,
                "queries_used": env._queries_used,
                "elapsed_s": round(elapsed, 1),
                "submitted_service": submitted_service,
                "true_service": true_service,
                "service_correct": submitted_service.lower() == true_service.lower(),
                "submitted_cause": submitted_cause[:100],
                "true_cause": true_cause[:100],
                "tool_sequence": [tc["tool"] for tc in tool_calls],
            }
            if verbose:
                print(f"    → reward={obs.reward:.4f} service_correct={result['service_correct']} ({elapsed:.1f}s)")
            return result

        result_text = format_tool_result(obs)
        messages.append({"role": "user", "content": f"Tool result:\n{result_text}"})

    return {"reward": 0.0, "error": "max_turns", "steps": 20}


def main():
    # Models to benchmark
    models = {
        "openai": {
            "client": OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
            "models": ["gpt-4o-mini", "gpt-4o", "o3-mini"],
        },
        "gemini": {
            "client": OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            "models": ["gemini-2.0-flash", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-03-25"],
        },
    }

    env = SREIncidentEnvironment()

    # Select scenarios across difficulties
    test_scenarios = [
        ("redis_cache_node_failure_001", "easy"),
        ("worker_oom_memory_leak_001", "medium"),
        ("payment_gateway_rate_limit_001", "medium"),
        ("n_plus_one_cache_bypass_001", "hard"),
        ("config_drift_gc_cascade_001", "expert"),
    ]

    results = {}

    print(f"{'='*80}")
    print(f"SRE INCIDENT ENV — FRONTIER MODEL BENCHMARK")
    print(f"{'='*80}")
    print(f"Scenarios: {len(test_scenarios)} | Timestamp: {datetime.now().isoformat()}")
    print()

    for provider, config in models.items():
        client = config["client"]
        for model_name in config["models"]:
            model_key = f"{provider}/{model_name}"
            results[model_key] = []
            print(f"\n{'─'*60}")
            print(f"MODEL: {model_key}")
            print(f"{'─'*60}")

            for scenario_id, difficulty in test_scenarios:
                print(f"\n  [{difficulty.upper()}] {scenario_id}:")
                result = run_episode(env, client, model_name, scenario_id, difficulty)
                result["scenario_id"] = scenario_id
                result["difficulty"] = difficulty
                results[model_key].append(result)

    # Summary table
    print(f"\n\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Model':<45} {'Avg Reward':>10} {'Svc Acc':>8} {'Avg Steps':>10}")
    print(f"{'─'*45} {'─'*10} {'─'*8} {'─'*10}")

    for model_key, model_results in results.items():
        valid = [r for r in model_results if "error" not in r]
        if not valid:
            print(f"{model_key:<45} {'ERROR':>10}")
            continue
        avg_reward = sum(r["reward"] for r in valid) / len(valid)
        svc_acc = sum(1 for r in valid if r.get("service_correct")) / len(valid)
        avg_steps = sum(r["steps"] for r in valid) / len(valid)
        print(f"{model_key:<45} {avg_reward:>10.4f} {svc_acc:>7.0%} {avg_steps:>10.1f}")

    # Per-difficulty breakdown
    print(f"\n\nPER-DIFFICULTY BREAKDOWN")
    print(f"{'─'*80}")
    for diff in ["easy", "medium", "hard", "expert"]:
        print(f"\n  {diff.upper()}:")
        for model_key, model_results in results.items():
            diff_results = [r for r in model_results if r.get("difficulty") == diff and "error" not in r]
            if diff_results:
                avg_r = sum(r["reward"] for r in diff_results) / len(diff_results)
                svc = sum(1 for r in diff_results if r.get("service_correct")) / len(diff_results)
                print(f"    {model_key:<40} reward={avg_r:.4f}  svc_acc={svc:.0%}")

    # Save raw results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nFull results saved to benchmark_results.json")


if __name__ == "__main__":
    main()
