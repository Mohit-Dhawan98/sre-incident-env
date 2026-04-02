"""V4 Benchmark — top OpenAI + Gemini models across all difficulties."""
import json, os, sys, asyncio, time
sys.stdout.reconfigure(line_buffering=True)
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
from server.environment import SREIncidentEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from inference import run_episode, mcp_tools_to_openai

class LocalAsyncWrapper:
    def __init__(self, env):
        self._env = env
    async def reset(self, **kw):
        return self._env.reset(**kw)
    async def step(self, action, **kw):
        return self._env.step(action, **kw)
    async def list_tools(self):
        return self._env.step(ListToolsAction()).tools
    async def close(self):
        pass

SCENARIOS = [
    # Easy (1)
    ("redis_cache_node_failure_001", "easy"),
    # Medium (3)
    ("worker_oom_memory_leak_001", "medium"),
    ("payment_gateway_rate_limit_001", "medium"),
    ("n_plus_one_cache_bypass_001", "medium"),
    # Hard (3)
    ("config_drift_gc_cascade_001", "hard"),
    ("slow_external_api_vendor_001", "hard"),
    ("disk_full_wal_corruption_001", "hard"),
    # Expert (3)
    ("kafka_partition_rebalance_storm_001", "expert"),
    ("memory_mapped_file_corruption_001", "expert"),
    ("tls_cert_chain_propagation_001", "expert"),
]

async def main():
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    gem = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                 base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    models = [
        ("o4-mini", oai, "o4-mini"),
        ("gpt-4o", oai, "gpt-4o"),
        ("gemini-2.5-flash", gem, "models/gemini-2.5-flash"),
    ]

    env_local = SREIncidentEnvironment()
    env_local.reset(difficulty="easy")
    tools = mcp_tools_to_openai(env_local.step(ListToolsAction()).tools)
    env = LocalAsyncWrapper(env_local)

    all_results = {}

    for label, client, mid in models:
        print(f"\n{'━'*60}")
        print(f"  {label}")
        print(f"{'━'*60}")
        all_results[label] = []

        for sid, diff in SCENARIOS:
            print(f"\n  [{diff.upper():>6}] {sid}:")
            env_local.reset(seed=42, difficulty=diff, scenario_id=sid)
            start = time.time()
            r = await run_episode(env, client, mid, tools, diff)
            elapsed = time.time() - start
            r["scenario"] = sid
            r["diff"] = diff
            r["elapsed"] = round(elapsed, 1)
            all_results[label].append(r)

    # Summary tables
    print(f"\n\n{'═'*80}")
    print("V4 BENCHMARK — FINAL RESULTS")
    print(f"{'═'*80}\n")

    # Per-scenario table
    print(f"{'Model':<18} {'Scenario':<45} {'Diff':>6} {'Score':>6} {'Time':>5}")
    print(f"{'─'*18} {'─'*45} {'─'*6} {'─'*6} {'─'*5}")
    for label, results in all_results.items():
        for r in results:
            score = f"{r['reward']:.3f}" if "error" not in r else "ERR"
            elapsed = f"{r.get('elapsed',0):.0f}s"
            print(f"{label:<18} {r['scenario']:<45} {r['diff']:>6} {score:>6} {elapsed:>5}")
        print()

    # Aggregate by difficulty
    print(f"\n{'Model':<18}", end="")
    for d in ["easy", "medium", "hard", "expert"]:
        print(f" {d.upper():>8}", end="")
    print(f" {'AVG':>8}")
    print(f"{'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for label, results in all_results.items():
        print(f"{label:<18}", end="")
        total, count = 0, 0
        for d in ["easy", "medium", "hard", "expert"]:
            d_results = [r for r in results if r["diff"] == d and "error" not in r]
            if d_results:
                avg = sum(r["reward"] for r in d_results) / len(d_results)
                print(f" {avg:>8.3f}", end="")
                total += sum(r["reward"] for r in d_results)
                count += len(d_results)
            else:
                print(f" {'ERR':>8}", end="")
        avg_all = total / count if count else 0
        print(f" {avg_all:>8.3f}")

    # Service accuracy by difficulty
    print(f"\nSERVICE ACCURACY")
    print(f"{'Model':<18}", end="")
    for d in ["easy", "medium", "hard", "expert"]:
        print(f" {d.upper():>8}", end="")
    print()
    print(f"{'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    # We need to check which service was submitted — extract from tool calls
    # For now just show scores > 0.25 as likely correct service
    for label, results in all_results.items():
        print(f"{label:<18}", end="")
        for d in ["easy", "medium", "hard", "expert"]:
            d_results = [r for r in results if r["diff"] == d and "error" not in r]
            if d_results:
                # Score > 0.25 means at least got the service right (0.25 for exact match)
                correct = sum(1 for r in d_results if r["reward"] >= 0.25)
                print(f" {correct}/{len(d_results):>5}", end="")
            else:
                print(f" {'?':>8}", end="")
        print()

    with open("benchmark_v4_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to benchmark_v4_results.json")

asyncio.run(main())
