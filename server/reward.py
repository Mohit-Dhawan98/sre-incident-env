"""Reward computation for SRE incident diagnosis — V4 (ungameable + learnable).

Design principles:
  - Every point requires genuine understanding. No free points.
  - Tier 1 (Diagnosis, 0.70): Components 2-4 GATED on correct service ID.
  - Tier 2 (Investigation, 0.30): NOT gated — provides RL gradient even on
    failed episodes. Measures investigation process quality.
  - Adjacency partial credit on service ID gives RL signal for "close but
    not quite — trace one more hop upstream."
  - No difficulty multiplier — harder scenarios are inherently harder.

Reward components (6, totaling 1.0):

  Tier 1 — Diagnosis Quality:
    1. Root service ID       (0.25): exact match + adjacency partial credit
    2. Failure type match    (0.15): exact match, GATED on service correct
    3. Root cause explanation (0.20): embedding sim, GATED on service correct
    4. Causal chain validity  (0.10): edge-fraction, GATED on service correct

  Tier 2 — Investigation Quality:
    5. Investigation efficiency (0.15): penalizes waste, NOT gated
    6. Investigation breadth    (0.15): rewards relevant coverage, NOT gated

Uses all-mpnet-base-v2 for semantic similarity — runs locally, no API calls.
"""

from typing import Any, Dict, List, Set, Tuple

from sentence_transformers import SentenceTransformer, util

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _model


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
) -> float:
    """Compute ungameable + learnable reward for incident diagnosis.

    Returns float in [0.0, 1.0].
    """

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
        submitted_text = submitted_root_cause.strip()
        if not submitted_text:
            semantic_score = 0.0
        else:
            model = _get_model()
            emb_sub = model.encode(submitted_text, convert_to_tensor=True)
            emb_true = model.encode(true_root_cause, convert_to_tensor=True)
            sim = float(util.cos_sim(emb_sub, emb_true).item())

            # Map [0.60, 1.0] → [0, 1] with power curve
            raw = max(0.0, (sim - 0.60) / 0.40)
            curved = raw ** 1.5

            # Length penalty: too short = no explanation, too long = kitchen sink
            text_len = len(submitted_text)
            if text_len < 20:
                length_factor = 0.0
            elif text_len > 500:
                length_factor = max(0.3, 500.0 / text_len)
            else:
                length_factor = 1.0

            semantic_score = curved * length_factor * 0.20

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
                    chain_score = f1 * 0.10
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
                        chain_score = (valid_edges / total_edges) * 0.10
                    else:
                        chain_score = 0.10

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
        efficiency_score = 0.15 * max(0.0, 1.0 - waste_fraction)

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

        breadth_score = 0.15 * relevance_ratio * spray_penalty
    else:
        breadth_score = 0.0

    # ══════════════════════════════════════════════════════════════════
    # TOTAL
    # ══════════════════════════════════════════════════════════════════

    total = (
        service_score
        + type_score
        + semantic_score
        + chain_score
        + efficiency_score
        + breadth_score
    )

    return round(min(1.0, max(0.0, total)), 4)
