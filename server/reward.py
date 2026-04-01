"""Reward computation for SRE incident diagnosis — V3 (ungameable).

Design principles:
  - Every point requires genuine understanding. No free points.
  - Components 2-4 are GATED on correct service identification.
    An agent that identifies the wrong service scores 0.0 (except the
    ~1/N random chance of guessing the service correctly).
  - No coverage, efficiency, or calibration components. The query budget
    is a constraint (enforced by the environment), not a reward signal.
  - Semantic similarity uses a high threshold (0.60) with a power curve
    to prevent vague descriptions from scoring partial credit.

Reward components (4, totaling 1.0):
    1. Service identification  (0.30): exact match on root service
    2. Failure type match      (0.20): exact match, GATED on #1
    3. Root cause explanation   (0.35): embedding similarity, GATED on #1
    4. Causal chain validity    (0.15): graph-validated chain, GATED on #1

Total max reward: 1.0, all scores clamped to [0.0, 1.0].
Zero LLM calls — only local embedding model for semantic similarity.
Deterministic given same inputs (embedding model is deterministic).
"""

from typing import Any, Dict, List, Set

from sentence_transformers import SentenceTransformer, util

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("google/embeddinggemma-300m")
    return _model


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
) -> float:
    """Compute ungameable reward for incident diagnosis.

    Args:
        submitted_root_cause: Agent's free-text root cause explanation.
        submitted_service: Agent's identified root cause service.
        true_root_cause: Ground truth root cause statement.
        true_root_service: Ground truth root cause service name.
        steps_used: Number of queries the agent used.
        query_budget: Maximum queries allowed.
        difficulty: Scenario difficulty level.
        confidence: Agent's self-reported confidence (unused in V3).
        services_queried: Set of service names the agent queried.
        all_services: Set of all service names in the scenario.
        submitted_failure_type: Agent's failure type classification.
        true_failure_type: Ground truth failure type.
        submitted_chain: Agent's causal chain (ordered list of services).
        services_graph: Dict of service_name -> {upstream: [list]} from scenario.
            Used to validate causal chain edges against real dependencies.

    Returns:
        Float reward in [0.0, 1.0].
    """

    # ── Component 1: Service Identification (0.30) ──────────────────────
    # Binary exact match. Cannot be gamed. Random guessing yields ~0.05.
    service_correct = (
        submitted_service.strip().lower() == true_root_service.strip().lower()
    )
    service_score = 0.30 if service_correct else 0.0

    # ── GATE: All remaining components require correct service ID ────────
    if not service_correct:
        return round(service_score, 4)  # 0.0 in practice

    # ── Component 2: Failure Type Classification (0.20) ─────────────────
    # Binary exact match, gated on service. Random guessing on type alone
    # yields 0.20/20 = 0.01, but since it's gated on service (0.05 random),
    # combined random expected value is ~0.0005. Effectively zero.
    type_score = (
        0.20
        if (
            submitted_failure_type.strip().lower()
            == true_failure_type.strip().lower()
            and submitted_failure_type.strip()
        )
        else 0.0
    )

    # ── Component 3: Root Cause Explanation (0.35) ──────────────────────
    # Embedding cosine similarity with:
    #   - High threshold (0.60) to reject vague descriptions
    #   - Power curve (exponent 1.5) to disproportionately reward precision
    #   - Length penalty to prevent kitchen-sink descriptions
    submitted_text = submitted_root_cause.strip()
    if not submitted_text:
        semantic_score = 0.0
    else:
        model = _get_model()
        emb_submitted = model.encode(submitted_text, convert_to_tensor=True)
        emb_true = model.encode(true_root_cause, convert_to_tensor=True)
        sim = float(util.cos_sim(emb_submitted, emb_true).item())

        # Map [0.60, 1.0] -> [0, 1] with power curve
        raw = max(0.0, (sim - 0.60) / 0.40)
        curved = raw ** 1.5

        # Length penalty: too short = no explanation, too long = kitchen sink
        text_len = len(submitted_text)
        if text_len < 20:
            length_factor = 0.0
        elif text_len > 500:
            # Graceful degradation, floor at 0.3
            length_factor = max(0.3, 500.0 / text_len)
        else:
            length_factor = 1.0

        semantic_score = curved * length_factor * 0.35

    # ── Component 4: Causal Chain Validity (0.15) ───────────────────────
    # All-or-nothing: chain must start with root service, every consecutive
    # pair must be a real dependency edge, and no duplicates.
    # If services_graph is unavailable, fall back to basic validation.
    chain_score = 0.0
    if submitted_chain and len(submitted_chain) >= 2:
        chain_lower = [s.strip().lower() for s in submitted_chain]
        all_lower = {s.lower() for s in all_services}

        starts_with_root = chain_lower[0] == true_root_service.strip().lower()
        all_real = all(s in all_lower for s in chain_lower)
        no_dupes = len(chain_lower) == len(set(chain_lower))

        if starts_with_root and all_real and no_dupes:
            if services_graph:
                # Validate each edge against the dependency graph.
                # chain[i] -> chain[i+1] means chain[i+1] calls chain[i],
                # so chain[i] should appear in chain[i+1]'s upstream list.
                # Build case-insensitive lookup.
                graph_lower = {}
                for svc_name, svc_info in services_graph.items():
                    upstream = svc_info.get("upstream", [])
                    graph_lower[svc_name.lower()] = [
                        u.lower() for u in upstream
                    ]

                total_edges = len(chain_lower) - 1
                valid_edges = 0
                for i in range(total_edges):
                    svc_from = chain_lower[i]
                    svc_to = chain_lower[i + 1]
                    # svc_to's upstream should contain svc_from
                    # (meaning svc_to depends on svc_from)
                    upstream_of_to = graph_lower.get(svc_to, [])
                    if svc_from in upstream_of_to:
                        valid_edges += 1

                # All-or-nothing: every edge must be valid
                if valid_edges == total_edges:
                    chain_score = 0.15
            else:
                # No graph available — fall back to basic validation only.
                # This is weaker but still requires root + real + no dupes.
                chain_score = 0.15

    # ── Total ───────────────────────────────────────────────────────────
    total = service_score + type_score + semantic_score + chain_score
    return round(min(1.0, max(0.0, total)), 4)
