"""Scenario parameter randomization for anti-memorization.

Implements domain randomization (Tier 3 per Procgen/LLF-Bench taxonomy):
surface features (service names, config values, version strings, timestamps)
are randomized per reset while the structural pattern (state-graph topology,
optimal path, trap placement) remains invariant.

This prevents agents from memorizing instance-specific values ("always call
zookeeper with session_timeout_ms=30000") and forces them to learn the
investigation + remediation pattern instead.

Architecture: name-pool substitution
- Each scenario defines a `template_vars` block mapping concrete names to
  pools of alternatives
- On reset(seed≠0), the resolver picks one value per pool using a seeded RNG
  and recursively walks the scenario dict, replacing both dict keys and
  string contents
- seed=0 returns the original scenario unchanged (backward compat)
- Domain-locked names (zookeeper, postgres, etcd, etc.) are NOT in the pools
  and therefore never change
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Union


def resolve_scenario(raw_scenario: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """Apply parameter randomization to a scenario.

    Args:
        raw_scenario: The scenario dict as loaded from JSONL (with template_vars).
        seed: Randomization seed.
              - None or 0: return the original scenario unchanged (backward compat).
              - Any positive int: deterministic randomized variant.

    Returns:
        A deep copy of the scenario with all template_vars substituted.
        The template_vars key itself is stripped from the output.
    """
    if seed is None or seed == 0:
        # Backward compat: return original scenario as-is (deep copy to prevent mutation)
        result = copy.deepcopy(raw_scenario)
        result.pop("template_vars", None)
        return result

    template_vars = raw_scenario.get("template_vars", {})
    if not template_vars:
        # No template vars defined — return unchanged
        result = copy.deepcopy(raw_scenario)
        result.pop("template_vars", None)
        return result

    # Build substitution map: old_name → sampled_new_name
    rng = random.Random(seed)
    subs: Dict[str, str] = {}
    for old_name, pool in template_vars.items():
        if isinstance(pool, list) and len(pool) > 0:
            subs[old_name] = str(rng.choice(pool))
        elif isinstance(pool, dict):
            # Support range specs: {"min": 4000, "max": 8000}
            if "min" in pool and "max" in pool:
                subs[old_name] = str(rng.randint(int(pool["min"]), int(pool["max"])))
            elif "choices" in pool:
                subs[old_name] = str(rng.choice(pool["choices"]))
        # If pool is empty or unrecognized, skip (leave original)

    if not subs:
        result = copy.deepcopy(raw_scenario)
        result.pop("template_vars", None)
        return result

    # Sort by longest key first to prevent partial-match collisions
    # e.g., "api-gateway" should be replaced before "api" (if both existed)
    sorted_subs = dict(sorted(subs.items(), key=lambda x: -len(x[0])))

    # Deep walk and substitute
    resolved = _deep_substitute(copy.deepcopy(raw_scenario), sorted_subs)
    resolved.pop("template_vars", None)
    return resolved


def _deep_substitute(obj: Any, subs: Dict[str, str]) -> Any:
    """Recursively walk a dict/list, substituting strings and dict keys."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_k = _sub_str(k, subs) if isinstance(k, str) else k
            new_dict[new_k] = _deep_substitute(v, subs)
        return new_dict
    elif isinstance(obj, list):
        return [_deep_substitute(item, subs) for item in obj]
    elif isinstance(obj, str):
        return _sub_str(obj, subs)
    else:
        return obj


def _sub_str(s: str, subs: Dict[str, str]) -> str:
    """Replace all occurrences of substitution keys in a string."""
    for old, new in subs.items():
        s = s.replace(old, new)
    return s


def verify_resolution(
    original: Dict[str, Any],
    resolved: Dict[str, Any],
    domain_locked: Optional[set] = None,
) -> List[str]:
    """Validate that a resolved scenario is structurally sound.

    Returns a list of issues (empty = all good).
    """
    issues: List[str] = []

    # 1. template_vars should be stripped
    if "template_vars" in resolved:
        issues.append("template_vars not stripped from resolved output")

    # 2. State graph topology must be invariant
    orig_states = set(original.get("failure", {}).get("remediation", {}).get("states", {}).keys())
    res_states = set(resolved.get("failure", {}).get("remediation", {}).get("states", {}).keys())
    if orig_states != res_states:
        issues.append(f"state names changed: orig={orig_states} resolved={res_states}")

    # 3. optimal_steps must be invariant
    orig_steps = original.get("failure", {}).get("remediation", {}).get("optimal_steps")
    res_steps = resolved.get("failure", {}).get("remediation", {}).get("optimal_steps")
    if orig_steps != res_steps:
        issues.append(f"optimal_steps changed: {orig_steps} → {res_steps}")

    # 4. All services referenced in state machine actions must exist in services dict
    res_services = set(resolved.get("services", {}).keys())
    for state_name, state_def in resolved.get("failure", {}).get("remediation", {}).get("states", {}).items():
        for action in state_def.get("actions", []):
            target = action.get("target", "")
            if target and target not in res_services:
                issues.append(f"state '{state_name}' action targets '{target}' which is not in services dict")

    # 5. Domain-locked names must still be present
    if domain_locked:
        for locked_name in domain_locked:
            if locked_name in original.get("services", {}) and locked_name not in res_services:
                issues.append(f"domain-locked service '{locked_name}' was removed by substitution")

    # 6. root_service must exist in services dict
    root_svc = resolved.get("failure", {}).get("root_service", "")
    if root_svc and root_svc not in res_services:
        issues.append(f"root_service '{root_svc}' not in services dict")

    return issues
