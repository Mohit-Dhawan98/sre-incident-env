"""Reward computation for SRE incident diagnosis — V5 (signal-quality first).

Design principles:
  - Weight proportional to signal quality. Clean signals get more weight.
  - Tier 1 components GATED on correct service ID (wrong service = 0 on Tier 1).
  - Tier 2 NOT gated — provides RL gradient even on failed episodes.
  - All signals are deterministic. No embedding models needed.
  - Explanation uses keyword checklist (golden data), not free-text similarity.

Reward components (7, totaling 1.0):

  Tier 1 — Did you solve it? (0.65):
    1. Root service ID        (0.25): exact match + adjacency(0.08) partial
    2. Failure type match     (0.15): exact match from taxonomy, GATED
    3. Causal chain validity   (0.15): F1 vs golden chain, GATED
    4. Explanation keywords    (0.10): golden keyword checklist, GATED

  Tier 2 — How did you solve it? (0.35):
    5. Query efficiency        (0.20): actual vs optimal queries, NOT gated
    6. Waste penalty           (0.08): duplicates/tunnel vision, NOT gated
    7. Investigation breadth   (0.07): queried causal chain services, NOT gated

All deterministic. No model loading. Tests run in <2s.
"""

from typing import Any, Dict, List, Set, Tuple

# Embedding model commented out — using pure Jaccard for explanation grading.
# Jaccard is more precise: rewards specific keywords, fully deterministic,
# no model loading, no GPU dependency.
# To re-enable: uncomment and set embedding_score weight in Component 3.
#
# from sentence_transformers import SentenceTransformer, util
# _model = None
# def _get_model() -> SentenceTransformer:
#     global _model
#     if _model is None:
#         _model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
#     return _model


def _get_neighbors(service: str, services_graph: Dict[str, Any]) -> Set[str]:
    """Get services one hop away in the dependency graph."""
    if not services_graph:
        return set()
    svc_lower = service.strip().lower()
    graph_lower = {}
    for name, info in services_graph.items():
        graph_lower[name.lower()] = [u.lower() for u in info.get("upstream", [])]

    neighbors = set()
    for name, upstreams in graph_lower.items():
        if name == svc_lower:
            # Services that call this one (upstreams)
            neighbors.update(upstreams)
        if svc_lower in upstreams:
            # Services this one calls (downstreams)
            neighbors.add(name)
    return neighbors


def _get_causal_chain_services(
    root_service: str, services_graph: Dict[str, Any]
) -> Set[str]:
    """Derive services in the failure propagation path via BFS from root.

    Failure propagates upstream: root fails → its callers are affected
    → their callers are affected, etc.

    In our graph, svc.upstream = [list of services that CALL this svc].
    So product-cache.upstream = [checkout-service] means checkout calls product-cache.
    When product-cache fails, checkout-service is affected (its dependency broke).
    """
    if not services_graph:
        return {root_service.lower()}

    graph_lower = {}
    for name, info in services_graph.items():
        graph_lower[name.lower()] = [u.lower() for u in info.get("upstream", [])]

    # "upstream" = callers of this service. When a service fails,
    # its callers are affected. So failure propagates TO the upstream list.
    root_lower = root_service.strip().lower()
    visited = set()
    queue = [root_lower]
    while queue:
        svc = queue.pop(0)
        if svc in visited:
            continue
        visited.add(svc)
        # This service's callers (upstream) are affected by its failure
        for caller in graph_lower.get(svc, []):
            if caller not in visited:
                queue.append(caller)

    return visited


def compute_reward(
    submitted_root_cause: str,
    submitted_service: str,
    true_root_cause: str,
    true_root_service: str,
    steps_used: int,
    query_budget: int,
    difficulty: str,
    confidence: float = 0.5,
    services_queried: Set[str] = frozenset(),
    all_services: Set[str] = frozenset(),
    submitted_failure_type: str = "",
    true_failure_type: str = "",
    submitted_chain: List[str] = (),
    services_graph: Dict[str, Any] = None,
    tool_call_history: List[Tuple[str, str]] = (),
    true_causal_chain: List[str] = (),
    optimal_queries: int = 10,
    explanation_keywords: List[str] = (),
    # V2 remediation params (optional — V1 scenarios pass defaults)
    system_healthy: bool = False,
    remediation_attempts: int = 0,
    harm_count: int = 0,
    correct_fix_used: bool = False,
    first_try_correct: bool = False,
    # V2.1 maze navigation params
    optimal_steps: int = 0,  # 0 = V1 mode (no remediation). >0 = V2 maze mode
    remediation_count: int = 0,
    observation_after_fix: int = 0,
    discovered_before_action: int = 0,
    execute_runbook_count: int = 0,
    unique_remediation_count: int = 0,
) -> float:
    """Compute reward for incident diagnosis + remediation maze navigation.

    V1 mode (no remediation data): Uses original 7-component scoring.
    V2 mode: Maze-based reward — how well did the agent navigate the state graph?

    Returns float in [0.0, 1.0].
    """
    # Detect V2 mode: if any remediation param is non-default, use V2 scoring
    # V2 mode detection: V2 callers (verify_resolution) explicitly pass optimal_steps > 0
    # V1 callers (submit_diagnosis) use the default optimal_steps=0
    is_v2 = optimal_steps > 0

    # ══════════════════════════════════════════════════════════════════
    # TIER 1: DIAGNOSIS QUALITY (0.70)
    # ══════════════════════════════════════════════════════════════════

    # ── Component 1: Root Service Identification (0.25) ──────────────
    # Exact match = 0.25. One-hop adjacent = 0.08. Else 0.0.
    sub_svc = submitted_service.strip().lower()
    true_svc = true_root_service.strip().lower()

    if sub_svc == true_svc:
        service_score = 0.25
        service_correct = True
    elif services_graph and sub_svc in _get_neighbors(true_svc, services_graph):
        service_score = 0.08
        service_correct = False  # Adjacent does NOT unlock gated components
    else:
        service_score = 0.0
        service_correct = False

    # ── GATE: Components 2-4 require EXACT service match ─────────────
    if not service_correct:
        # Skip to Tier 2 (investigation quality — not gated)
        type_score = 0.0
        semantic_score = 0.0
        chain_score = 0.0
    else:
        # ── Component 2: Failure Type Classification (0.15) ──────────
        type_score = (
            0.15
            if (
                submitted_failure_type.strip().lower()
                == true_failure_type.strip().lower()
                and submitted_failure_type.strip()
            )
            else 0.0
        )

        # ── Component 3: Root Cause Explanation (0.20) ───────────────
        # Keyword checklist: golden data defines specific keywords the
        # explanation must contain (root service, mechanism words, downstream).
        # Score = fraction of keywords found in submission.
        # Clean signal: each keyword is deliberately chosen, no noise.
        submitted_text = submitted_root_cause.strip().lower()
        if not submitted_text or len(submitted_text) < 10:
            semantic_score = 0.0
        elif explanation_keywords:
            found = 0
            for keyword in explanation_keywords:
                if keyword.lower() in submitted_text:
                    found += 1
            semantic_score = (found / len(explanation_keywords)) * 0.10
        else:
            # Fallback to simple Jaccard if no checklist defined
            stopwords = {'the', 'and', 'was', 'for', 'that', 'with', 'from',
                         'this', 'are', 'were', 'been', 'has', 'had', 'not',
                         'but', 'all', 'its', 'due', 'causing', 'caused'}
            def extract_kw(text):
                return {w.strip('.,;:()[]"\'') for w in text.lower().split()
                        if len(w.strip('.,;:()[]"\'')) >= 3 and w.strip('.,;:()[]"\'') not in stopwords}
            true_kw = extract_kw(true_root_cause)
            sub_kw = extract_kw(submitted_text)
            union = len(true_kw | sub_kw)
            semantic_score = (len(true_kw & sub_kw) / union * 0.10) if union else 0.0

        # ── Component 4: Causal Chain Validity (0.10) ────────────────
        # Compared against golden causal_chain from scenario.
        # Score = overlap between submitted chain and true chain.
        chain_score = 0.0
        if submitted_chain and len(submitted_chain) >= 1:
            chain_lower = [s.strip().lower() for s in submitted_chain]

            if true_causal_chain:
                # Compare against golden chain
                true_chain_lower = [s.strip().lower() for s in true_causal_chain]
                true_chain_set = set(true_chain_lower)

                starts_with_root = chain_lower[0] == true_svc
                # What fraction of the true chain did the agent capture?
                submitted_set = set(chain_lower)
                overlap = len(submitted_set & true_chain_set)
                precision = overlap / len(submitted_set) if submitted_set else 0
                recall = overlap / len(true_chain_set) if true_chain_set else 0

                if starts_with_root and precision > 0 and recall > 0:
                    # F1-like score: harmonic mean of precision and recall
                    f1 = 2 * precision * recall / (precision + recall)
                    chain_score = f1 * 0.15
            elif services_graph:
                # Fallback: validate edges against graph
                all_lower = {s.lower() for s in all_services}
                starts_with_root = chain_lower[0] == true_svc
                all_real = all(s in all_lower for s in chain_lower)
                no_dupes = len(chain_lower) == len(set(chain_lower))
                if starts_with_root and all_real and no_dupes:
                    graph_lower = {}
                    for svc_name, svc_info in services_graph.items():
                        graph_lower[svc_name.lower()] = [
                            u.lower() for u in svc_info.get("upstream", [])
                        ]
                    total_edges = len(chain_lower) - 1
                    valid_edges = 0
                    for i in range(total_edges):
                        upstream_of_from = graph_lower.get(chain_lower[i], [])
                        if chain_lower[i + 1] in upstream_of_from:
                            valid_edges += 1
                    if total_edges > 0:
                        chain_score = (valid_edges / total_edges) * 0.15
                    else:
                        chain_score = 0.15

    # ══════════════════════════════════════════════════════════════════
    # TIER 2: INVESTIGATION QUALITY (0.30) — NOT gated on diagnosis
    # ══════════════════════════════════════════════════════════════════

    # ── Component 5: Investigation Efficiency (0.15) ─────────────────
    # Penalizes: duplicate queries, nonexistent service queries, tunnel vision.
    # Requires minimum 2 investigative queries (prevents "guess immediately").
    investigative_calls = [
        (t, a) for t, a in tool_call_history
        if t in ("read_logs", "check_metric")
    ]
    num_investigative = len(investigative_calls)

    if num_investigative < 2:
        # Rushed — didn't investigate, just guessed
        efficiency_score = 0.0
    else:
        wasted = 0
        seen_calls = set()
        consecutive_same_service = 0
        prev_service = None

        for tool_name, service_arg in investigative_calls:
            call_key = (tool_name, service_arg.lower() if service_arg else "")

            # Duplicate detection
            if call_key in seen_calls:
                wasted += 1
            seen_calls.add(call_key)

            # Nonexistent service detection
            if service_arg and service_arg.lower() not in {s.lower() for s in all_services}:
                wasted += 1

            # Tunnel vision: 3+ consecutive queries to same service
            svc = (service_arg or "").lower()
            if svc == prev_service:
                consecutive_same_service += 1
                if consecutive_same_service >= 2:  # 3rd+ consecutive
                    wasted += 1
            else:
                consecutive_same_service = 0
            prev_service = svc

        waste_fraction = wasted / num_investigative if num_investigative > 0 else 0
        efficiency_score = 0.08 * max(0.0, 1.0 - waste_fraction)

    # ── Component 6: Investigation Breadth (0.15) ────────────────────
    # Rewards querying services in the causal chain. Penalizes spray-and-pray.
    if services_queried:
        # Use golden causal chain if available, else BFS fallback
        if true_causal_chain:
            causal_services = {s.lower() for s in true_causal_chain}
        elif services_graph:
            causal_services = _get_causal_chain_services(true_root_service, services_graph)
        else:
            causal_services = {true_root_service.lower()}
        queried_lower = {s.lower() for s in services_queried}

        # How many causal-chain services did you investigate?
        relevant_queried = len(queried_lower & causal_services)
        total_causal = len(causal_services)

        if total_causal > 0:
            relevance_ratio = min(1.0, relevant_queried / total_causal)
        else:
            relevance_ratio = 0.0

        # Penalize spray-and-pray: querying >2x the causal chain services
        over_query_ratio = len(queried_lower) / max(1, total_causal)
        if over_query_ratio > 2.0:
            spray_penalty = max(0.5, 2.0 / over_query_ratio)
        else:
            spray_penalty = 1.0

        breadth_score = 0.07 * relevance_ratio * spray_penalty
    else:
        breadth_score = 0.0

    # ── Component 7: Query Efficiency (0.15) ───────────────────────
    # Compares actual queries used vs optimal (golden data per scenario).
    # Full credit at <= 1.5x optimal. Linear decay to 3x. Zero beyond.
    # Also zero if < 3 queries (rushed/guessed).
    total_queries = len([t for t in tool_call_history if t[0] in ("read_logs", "check_metric")])
    if total_queries < 3:
        query_efficiency_score = 0.0  # Rushed — didn't investigate
    elif optimal_queries > 0:
        ratio = total_queries / optimal_queries
        if ratio <= 1.5:
            query_efficiency_score = 0.20  # Within sweet spot
        elif ratio <= 3.0:
            # Linear decay from 1.5x to 3x
            query_efficiency_score = 0.20 * (1.0 - (ratio - 1.5) / 1.5)
        else:
            query_efficiency_score = 0.0  # Excessive waste
    else:
        query_efficiency_score = 0.20

    # ══════════════════════════════════════════════════════════════════
    # TOTAL (V1 or V2)
    # ══════════════════════════════════════════════════════════════════

    if not is_v2:
        # V1 mode: original 7-component scoring (investigation + diagnosis only)
        total = (
            service_score
            + type_score
            + semantic_score
            + chain_score
            + efficiency_score
            + breadth_score
            + query_efficiency_score
        )
        return round(min(1.0, max(0.0, total)), 4)

    # ══════════════════════════════════════════════════════════════════
    # V2.1 MODE: MAZE NAVIGATION REWARD
    # ══════════════════════════════════════════════════════════════════
    #
    # Treat the incident as a maze. Score HOW the agent navigated.
    #
    # 1. REACHED EXIT (0.30) — did the system reach healthy?
    # 2. PATH EFFICIENCY (0.20) — optimal steps vs actual
    # 3. TRAP AVOIDANCE (0.15) — avoided worsened outcomes
    # 4. SCOUTING (0.10) — observed after fix + discovered before action
    # 5. NO WASTED MOVES (0.10) — unique actions vs total
    # 6. DIAGNOSIS (0.15) — understood what they fixed

    # ── 1. REACHED EXIT (0.45) ─────────────────────────────────────
    maze_exit = 0.45 if system_healthy else 0.0

    # ── 2. PATH EFFICIENCY (0.15) ──────────────────────────────────
    # How many remediation steps vs optimal?
    if remediation_count == 0:
        maze_efficiency = 0.0  # Never tried to fix
    elif remediation_count <= optimal_steps:
        maze_efficiency = 0.15  # At or below optimal
    elif remediation_count <= optimal_steps * 2:
        maze_efficiency = 0.15 * (1.0 - (remediation_count - optimal_steps) / optimal_steps)
    elif remediation_count <= optimal_steps * 4:
        maze_efficiency = 0.15 * 0.25 * (1.0 - (remediation_count - optimal_steps * 2) / (optimal_steps * 2))
    else:
        maze_efficiency = 0.0

    # ── 3. TRAP AVOIDANCE (0.10) ───────────────────────────────────
    # Start at 0.10, lose 0.05 per trap door triggered
    maze_traps = max(0.0, 0.10 - harm_count * 0.05)

    # ── 4. SCOUTING (0.10) — only counts if agent actually remediated ──
    # 4a. Observe after fix: read_logs/check_metric after remediation (0.05)
    if remediation_count > 0:
        observe_ratio = min(1.0, observation_after_fix / remediation_count)
        maze_observe = 0.05 * observe_ratio
    else:
        maze_observe = 0.0  # No remediation = no scouting credit

    # 4b. Discover before action: get_service_info before execute_runbook (0.05)
    if execute_runbook_count > 0:
        discover_ratio = min(1.0, discovered_before_action / execute_runbook_count)
        maze_discover = 0.05 * discover_ratio
    elif remediation_count > 0:
        maze_discover = 0.05  # Used platform tools only (no execute_runbook needed)
    else:
        maze_discover = 0.0  # No remediation = no scouting credit

    maze_scout = maze_observe + maze_discover

    # ── 5. NO WASTED MOVES (0.10) ──────────────────────────────────
    if remediation_count > 0:
        unique_ratio = min(1.0, unique_remediation_count / remediation_count)
        maze_unique = 0.10 * unique_ratio
    else:
        maze_unique = 0.0  # No remediation = no credit

    # ── 6. DIAGNOSIS (0.10) ────────────────────────────────────────
    # GATED on system_healthy — if you didn't fix it, diagnosis is worthless
    if system_healthy:
        sub_svc = submitted_service.strip().lower()
        true_svc = true_root_service.strip().lower()

        diag_service = 0.05 if sub_svc == true_svc else (0.02 if services_graph and sub_svc in _get_neighbors(true_svc, services_graph) else 0.0)

        diag_type = 0.03 if (submitted_failure_type.strip().lower() == true_failure_type.strip().lower() and submitted_failure_type.strip()) else 0.0

        submitted_text = submitted_root_cause.strip().lower()
        if explanation_keywords and len(submitted_text) >= 10:
            found = sum(1 for kw in explanation_keywords if kw.lower() in submitted_text)
            diag_keywords = (found / len(explanation_keywords)) * 0.02
        else:
            diag_keywords = 0.0

        maze_diagnosis = diag_service + diag_type + diag_keywords
    else:
        maze_diagnosis = 0.0  # Didn't fix it = diagnosis worthless

    # ══════════════════════════════════════════════════════════════════
    # TOTAL
    # ══════════════════════════════════════════════════════════════════
    total = (
        maze_exit
        + maze_efficiency
        + maze_traps
        + maze_scout
        + maze_unique
        + maze_diagnosis
    )

    return round(min(1.0, max(0.0, total)), 4)
