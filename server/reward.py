"""Reward computation for SRE incident diagnosis.

Uses EmbeddingGemma-300M for semantic similarity — runs locally, no API calls.
No difficulty multiplier — harder scenarios are inherently harder to score on.

Reward components (7):
    1. Service accuracy   (0.20): exact match on root service name
    2. Failure type match (0.15): exact match on failure category (deterministic)
    3. Semantic similarity (0.25): embedding cosine sim of root cause statements
    4. Causal chain score (0.10): overlap between submitted and true causal chain
    5. Investigation coverage (0.15): fraction of services investigated
    6. Efficiency          (0.10): non-linear — penalizes budget exhaustion only
    7. Confidence calibration (0.05): penalizes confident wrong answers

Total max reward: 1.0, all scores clamped to [0.0, 1.0].
"""

from typing import List, Set

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
) -> float:
    model = _get_model()

    # 1. Service identification (0 or 0.20)
    service_score = (
        0.20
        if submitted_service.strip().lower() == true_root_service.strip().lower()
        else 0.0
    )

    # 2. Failure type match (0 or 0.15) — deterministic
    type_score = (
        0.15
        if submitted_failure_type.strip().lower() == true_failure_type.strip().lower()
        and submitted_failure_type.strip()
        else 0.0
    )

    # 3. Semantic similarity of root cause statement (0 to 0.25)
    if not submitted_root_cause.strip():
        semantic_score = 0.0
    else:
        emb_submitted = model.encode(submitted_root_cause, convert_to_tensor=True)
        emb_true = model.encode(true_root_cause, convert_to_tensor=True)
        sim = float(util.cos_sim(emb_submitted, emb_true).item())
        semantic_score = max(0.0, (sim - 0.5) / 0.5) * 0.25

    # 4. Causal chain score (0 to 0.10)
    # Reward for including the root service in the chain + overlap with real services
    if submitted_chain:
        chain_set = set(s.lower() for s in submitted_chain)
        real_services = set(s.lower() for s in all_services)
        # Points for: root service in chain, chain services are real services
        root_in_chain = 1.0 if true_root_service.lower() in chain_set else 0.0
        if real_services:
            valid_ratio = len(chain_set & real_services) / len(chain_set) if chain_set else 0
        else:
            valid_ratio = 0.0
        chain_score = (root_in_chain * 0.5 + valid_ratio * 0.5) * 0.10
    else:
        chain_score = 0.0

    # 5. Investigation coverage (0 to 0.15)
    if all_services:
        queried_valid = services_queried & all_services
        coverage_ratio = len(queried_valid) / len(all_services)
        coverage_score = coverage_ratio * 0.15
    else:
        coverage_score = 0.0

    # 6. Non-linear efficiency (0 to 0.10)
    if query_budget > 0:
        usage_ratio = steps_used / query_budget
        if usage_ratio <= 0.7:
            efficiency_score = 0.10
        elif usage_ratio <= 1.0:
            efficiency_score = 0.10 * (1.0 - (usage_ratio - 0.7) / 0.3)
        else:
            efficiency_score = 0.0
    else:
        efficiency_score = 0.0

    # 7. Confidence calibration (0 to 0.05)
    deterministic_accuracy = service_score + type_score  # max 0.35
    accuracy_normalized = (deterministic_accuracy + semantic_score) / 0.60
    accuracy_normalized = min(1.0, accuracy_normalized)
    confidence = max(0.0, min(1.0, confidence))
    calibration_error = abs(confidence - accuracy_normalized)
    calibration_score = (1.0 - calibration_error) * 0.05

    total = (
        service_score
        + type_score
        + semantic_score
        + chain_score
        + coverage_score
        + efficiency_score
        + calibration_score
    )

    return round(min(1.0, max(0.0, total)), 4)
