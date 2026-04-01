"""Reward computation for SRE incident diagnosis.

Uses EmbeddingGemma-300M for semantic similarity — runs locally, no API calls.

Reward components (5):
    1. Service accuracy  (0.25): exact match on root service name
    2. Semantic similarity (0.45): embedding cosine sim of root cause statements
    3. Investigation coverage (0.15): fraction of services investigated
    4. Efficiency (0.10): non-linear — penalizes budget exhaustion, not thoroughness
    5. Confidence calibration (0.05): penalizes confident wrong answers

Total max reward: 1.0, all scores clamped to [0.0, 1.0].
"""

from typing import Set

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
) -> float:
    model = _get_model()

    # Component 1: Service identification (0 or 0.25)
    service_score = (
        0.25
        if submitted_service.strip().lower() == true_root_service.strip().lower()
        else 0.0
    )

    # Component 2: Semantic similarity of root cause statement (0 to 0.45)
    if not submitted_root_cause.strip():
        semantic_score = 0.0
    else:
        emb_submitted = model.encode(submitted_root_cause, convert_to_tensor=True)
        emb_true = model.encode(true_root_cause, convert_to_tensor=True)
        sim = float(util.cos_sim(emb_submitted, emb_true).item())
        semantic_score = max(0.0, (sim - 0.5) / 0.5) * 0.45

    # Component 3: Investigation coverage (0 to 0.15)
    if all_services:
        queried_valid = services_queried & all_services
        coverage_ratio = len(queried_valid) / len(all_services)
        coverage_score = coverage_ratio * 0.15
    else:
        coverage_score = 0.0

    # Component 4: Non-linear efficiency (0 to 0.10)
    # Full bonus if ≤70% budget used, gentle decline after, 0 if exhausted
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

    # Component 5: Confidence calibration (0 to 0.05)
    # Reward agents whose confidence matches their actual accuracy
    accuracy_normalized = (service_score + semantic_score) / 0.70
    accuracy_normalized = min(1.0, accuracy_normalized)
    confidence = max(0.0, min(1.0, confidence))
    calibration_error = abs(confidence - accuracy_normalized)
    calibration_score = (1.0 - calibration_error) * 0.05

    total = (
        service_score
        + semantic_score
        + coverage_score
        + efficiency_score
        + calibration_score
    )

    # Difficulty multiplier
    multipliers = {"easy": 0.6, "medium": 0.8, "hard": 1.0, "expert": 1.2}
    total = total * multipliers.get(difficulty, 1.0)

    return round(min(1.0, max(0.0, total)), 4)
