"""
Comprehensive benchmark of ALL frontier models on SRE Incident Response env.

Models tested:
  OpenAI: gpt-4o-mini, gpt-4o, o3-mini, o3, o4-mini
  Gemini: 2.0-flash, 2.5-flash, 2.5-pro, 3.1-flash (if available)
"""

import json
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

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

IMPORTANT: Respond with ONLY a JSON object. No markdown, no explanation.
{"tool": "list_services"}
{"tool": "read_logs", "args": {"service": "api-gateway", "window_minutes": 15, "level_filter": "ERROR"}}
{"tool": "submit_diagnosis", "args": {"root_cause": "db connection pool exhausted", "affected_service": "db-primary", "confidence": 0.85}}
"""


def parse_llm_action(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise


def format_tool_result(obs):
    if hasattr(obs, 'result') and obs.result and hasattr(obs.result, 'data'):
        return obs.result.data
    if hasattr(obs, 'metadata') and obs.metadata:
        return json.dumps(obs.metadata, indent=2)
    return str(obs)


def run_episode(env, llm, model_name, scenario_id, difficulty, verbose=True):
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
    # GPT-5.x and reasoning models use max_completion_tokens, not max_tokens
    is_reasoning = any(x in model_name for x in ["o3", "o1", "o4", "gpt-5"])

    for turn in range(20):
        try:
            kwargs = {"model": model_name, "messages": messages, "timeout": 60}
            if is_reasoning:
                kwargs["max_completion_tokens"] = 2000
            else:
                kwargs["temperature"] = 0.1
                kwargs["max_tokens"] = 500
            response = llm.chat.completions.create(**kwargs)
        except Exception as e:
            err_str = str(e)
            if verbose:
                print(f"    API error: {err_str[:120]}")
            return {"reward": 0.0, "error": err_str[:200], "steps": turn}

        assistant_text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_text})

        try:
            action = parse_llm_action(assistant_text)
        except (json.JSONDecodeError, ValueError):
            parse_errors += 1
            if parse_errors >= 3:
                return {"reward": 0.0, "error": "parse_failures", "steps": turn}
            messages.append({"role": "user", "content": "Invalid JSON. Respond with ONLY {\"tool\": \"...\"}"})
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
            messages.append({"role": "user", "content": f"Error: {e}. Try a different action."})
            continue

        if obs.done:
            elapsed = time.time() - start_time
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
                "submitted_cause": submitted_cause[:150],
                "true_cause": true_cause[:150],
                "tool_sequence": [tc["tool"] for tc in tool_calls],
            }
            if verbose:
                svc_mark = "✓" if result["service_correct"] else "✗"
                print(f"    → reward={obs.reward:.4f} svc={svc_mark} ({elapsed:.1f}s)")
            return result

        messages.append({"role": "user", "content": f"Tool result:\n{format_tool_result(obs)}"})

    return {"reward": 0.0, "error": "max_turns", "steps": 20}


def main():
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    gemini_client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    # All frontier CHAT models — April 2026
    # Excluded: gpt-5.4-pro (not a chat model), gpt-5-pro (not a chat model)
    model_configs = [
        # OpenAI GPT-5.x chat models
        ("gpt-5.4", openai_client, "gpt-5.4"),
        ("gpt-5.4-mini", openai_client, "gpt-5.4-mini"),
        ("gpt-5.2", openai_client, "gpt-5.2"),
        ("gpt-5.1", openai_client, "gpt-5.1"),
        ("gpt-5", openai_client, "gpt-5"),
        ("gpt-5-mini", openai_client, "gpt-5-mini"),
        # OpenAI codex models
        ("gpt-5.2-codex", openai_client, "gpt-5.2-codex"),
        ("gpt-5.1-codex", openai_client, "gpt-5.1-codex"),
        # OpenAI older
        ("gpt-4.1", openai_client, "gpt-4.1"),
        ("gpt-4o", openai_client, "gpt-4o"),
        ("gpt-4o-mini", openai_client, "gpt-4o-mini"),
        # OpenAI reasoning
        ("o4-mini", openai_client, "o4-mini"),
        ("o3", openai_client, "o3"),
        ("o3-mini", openai_client, "o3-mini"),
        # Gemini (newest → oldest)
        ("gemini-3.1-pro", gemini_client, "models/gemini-3.1-pro-preview"),
        ("gemini-3-pro", gemini_client, "models/gemini-3-pro-preview"),
        ("gemini-3-flash", gemini_client, "models/gemini-3-flash-preview"),
        ("gemini-2.5-pro", gemini_client, "models/gemini-2.5-pro"),
        ("gemini-2.5-flash", gemini_client, "models/gemini-2.5-flash"),
        ("gemini-2.0-flash", gemini_client, "models/gemini-2.0-flash"),
    ]

    env = SREIncidentEnvironment()

    test_scenarios = [
        ("redis_cache_node_failure_001", "easy"),
        ("worker_oom_memory_leak_001", "medium"),
        ("payment_gateway_rate_limit_001", "medium"),
        ("n_plus_one_cache_bypass_001", "hard"),
        ("config_drift_gc_cascade_001", "expert"),
    ]

    all_results = {}

    print(f"{'='*80}")
    print(f"SRE INCIDENT ENV — COMPREHENSIVE FRONTIER MODEL BENCHMARK")
    print(f"{'='*80}")
    print(f"Models: {len(model_configs)} | Scenarios: {len(test_scenarios)}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    for label, client, model_id in model_configs:
        all_results[label] = []
        print(f"\n{'━'*60}")
        print(f"  {label} ({model_id})")
        print(f"{'━'*60}")

        for scenario_id, difficulty in test_scenarios:
            print(f"\n  [{difficulty.upper()}] {scenario_id}:")
            result = run_episode(env, client, model_id, scenario_id, difficulty)
            result["scenario_id"] = scenario_id
            result["difficulty"] = difficulty
            all_results[label].append(result)

    # ═══════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════
    print(f"\n\n{'═'*80}")
    print("RESULTS SUMMARY")
    print(f"{'═'*80}")
    print(f"\n{'Model':<25} {'Avg Reward':>10} {'Svc Acc':>8} {'Avg Steps':>10} {'Errors':>7}")
    print(f"{'─'*25} {'─'*10} {'─'*8} {'─'*10} {'─'*7}")

    for label, results in all_results.items():
        valid = [r for r in results if "error" not in r]
        errors = len(results) - len(valid)
        if not valid:
            print(f"{label:<25} {'ALL ERR':>10} {'':>8} {'':>10} {errors:>7}")
            continue
        avg_reward = sum(r["reward"] for r in valid) / len(valid)
        svc_acc = sum(1 for r in valid if r.get("service_correct")) / len(valid)
        avg_steps = sum(r["steps"] for r in valid) / len(valid)
        print(f"{label:<25} {avg_reward:>10.4f} {svc_acc:>7.0%} {avg_steps:>10.1f} {errors:>7}")

    # Per-difficulty
    print(f"\n\nPER-DIFFICULTY BREAKDOWN")
    print(f"{'─'*80}")
    for diff in ["easy", "medium", "hard", "expert"]:
        print(f"\n  {diff.upper()}:")
        print(f"  {'Model':<25} {'Reward':>8} {'Svc':>5} {'Submitted Service':<25} {'True Service':<20}")
        print(f"  {'─'*25} {'─'*8} {'─'*5} {'─'*25} {'─'*20}")
        for label, results in all_results.items():
            for r in results:
                if r.get("difficulty") == diff and "error" not in r:
                    svc_mark = "✓" if r.get("service_correct") else "✗"
                    sub_svc = r.get("submitted_service", "?")[:24]
                    true_svc = r.get("true_service", "?")[:19]
                    print(f"  {label:<25} {r['reward']:>8.4f} {svc_mark:>5} {sub_svc:<25} {true_svc:<20}")

    # Diagnosis comparison
    print(f"\n\nDIAGNOSIS COMPARISON (what each model submitted)")
    print(f"{'─'*80}")
    for scenario_id, difficulty in test_scenarios:
        print(f"\n  [{difficulty.upper()}] {scenario_id}")
        # Find true cause
        for label, results in all_results.items():
            for r in results:
                if r.get("scenario_id") == scenario_id and "error" not in r:
                    print(f"  TRUE: {r.get('true_cause', '?')[:75]}")
                    break
            break
        for label, results in all_results.items():
            for r in results:
                if r.get("scenario_id") == scenario_id and "error" not in r:
                    print(f"  {label:<20}: {r.get('submitted_cause', '?')[:55]}  [r={r['reward']:.3f}]")

    # Save
    with open("benchmark_full_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nSaved to benchmark_full_results.json")


if __name__ == "__main__":
    main()
