"""Quick benchmark — top 5 models only."""
import json, os, sys, time
from dotenv import load_dotenv
sys.stdout.reconfigure(line_buffering=True)
load_dotenv()

from openai import OpenAI
from server.environment import SREIncidentEnvironment
from openenv.core.env_server.mcp_types import CallToolAction

SYSTEM_PROMPT = """You are an expert SRE investigating a production incident.
Tools (respond with JSON only):
- list_services: See services (free)
- read_logs: Args: service, window_minutes(5), level_filter(null)
- check_metric: Args: service, metric, window_minutes(10)
- submit_diagnosis: Args: root_cause, affected_service, confidence

Trace symptoms → root cause (2-3 hops deep). Watch for red herrings.
Respond ONLY with JSON: {"tool": "...", "args": {...}}"""

def parse(text):
    text = text.strip()
    if text.startswith("```"): text = "\n".join(text.split("\n")[1:-1]).strip()
    s, e = text.find("{"), text.rfind("}") + 1
    return json.loads(text[s:e]) if s >= 0 else json.loads(text)

def run(env, llm, model, sid, diff):
    obs = env.reset(seed=42, difficulty=diff, scenario_id=sid)
    true_svc = env._scenario["failure"]["root_service"]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs.metadata["message"]}]
    is_r = any(x in model for x in ["o3", "o4", "gpt-5"])
    for t in range(15):
        try:
            kw = {"model": model, "messages": msgs, "timeout": 45}
            if is_r: kw["max_completion_tokens"] = 1500
            else: kw["temperature"] = 0.1; kw["max_tokens"] = 400
            r = llm.chat.completions.create(**kw)
        except Exception as e:
            print(f"    ERR: {str(e)[:80]}")
            return {"reward": 0, "error": str(e)[:80]}
        txt = r.choices[0].message.content or ""
        msgs.append({"role": "assistant", "content": txt})
        try: a = parse(txt)
        except:
            msgs.append({"role": "user", "content": 'JSON only: {"tool":"list_services"}'})
            continue
        tn, ta = a.get("tool",""), a.get("args",{})
        print(f"    T{t+1}: {tn}", end="", flush=True)
        try: obs = env.step(CallToolAction(tool_name=tn, arguments=ta))
        except Exception as e:
            msgs.append({"role":"user","content":f"Error: {e}"})
            print(f" ERR"); continue
        if obs.done:
            ss = ta.get("affected_service","") if tn=="submit_diagnosis" else ""
            ok = "✓" if ss.lower()==true_svc.lower() else "✗"
            print(f" → {obs.reward:.3f} {ok}")
            return {"reward": float(obs.reward), "svc_correct": ss.lower()==true_svc.lower(),
                    "steps": t+1, "submitted": ta.get("root_cause","")[:80]}
        print()
        msgs.append({"role":"user","content":obs.result.data if hasattr(obs,'result') and obs.result else str(obs.metadata)})
    return {"reward": 0, "error": "max_turns"}

oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
gem = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
             base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

models = [
    ("gpt-5.4",     oai, "gpt-5.4"),
    ("gpt-5.1-codex", oai, "gpt-5.1-codex"),
    ("o4-mini",     oai, "o4-mini"),
    ("gpt-4o",      oai, "gpt-4o"),
    ("gemini-2.5-pro", gem, "models/gemini-2.5-pro"),
]

scenarios = [
    ("redis_cache_node_failure_001", "easy"),
    ("worker_oom_memory_leak_001", "medium"),
    ("n_plus_one_cache_bypass_001", "hard"),
    ("config_drift_gc_cascade_001", "expert"),
]

env = SREIncidentEnvironment()
results = {}

for label, client, mid in models:
    print(f"\n{'='*50}\n  {label}\n{'='*50}")
    results[label] = []
    for sid, diff in scenarios:
        print(f"\n  [{diff.upper()}] {sid}:")
        r = run(env, client, mid, sid, diff)
        r["scenario"] = sid; r["diff"] = diff
        results[label].append(r)

print(f"\n\n{'='*60}")
print(f"{'Model':<20} {'Easy':>6} {'Med':>6} {'Hard':>6} {'Expert':>6} {'Avg':>6} {'SvcAcc':>7}")
print(f"{'─'*20} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*7}")
for label, rs in results.items():
    vals = {r["diff"]: r.get("reward",0) for r in rs}
    valid = [r for r in rs if "error" not in r]
    avg = sum(r["reward"] for r in valid)/len(valid) if valid else 0
    sacc = sum(1 for r in valid if r.get("svc_correct"))/len(valid) if valid else 0
    print(f"{label:<20} {vals.get('easy',0):>6.3f} {vals.get('medium',0):>6.3f} {vals.get('hard',0):>6.3f} {vals.get('expert',0):>6.3f} {avg:>6.3f} {sacc:>6.0%}")

with open("benchmark_results_top5.json","w") as f: json.dump(results,f,indent=2,default=str)
print("\nSaved to benchmark_results_top5.json")
