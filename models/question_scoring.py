"""
Exploratory question scoring.

This layer wraps the existing EIG acquisition function with product/research
signals: anchor coverage, generated-probe exploration, semantic novelty, and
redundancy penalties. It does not change posterior math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from models.question_assignment import cosine_similarity_matrix, embed_texts
from models.question_bank import Question
from models.question_selection import expected_information_gain


DEFAULT_WEIGHTS = {
    "eig": 1.0,
    "novelty": 0.15,
    "exploration": 0.20,
    "policy": 0.05,
    "redundancy": 0.25,
    "risk": 1.0,
}


@dataclass(frozen=True)
class SessionComposition:
    """Hard composition limits for anchored exploratory sessions."""

    max_inference_questions: int
    min_anchor_questions: int = 2
    max_generated_probes: int = 6

    @classmethod
    def scaled(
        cls,
        max_inference_questions: int,
        min_anchor_questions: int = 2,
        max_generated_probes: int = 6,
    ) -> "SessionComposition":
        max_inf = max(1, int(max_inference_questions))
        anchors = max(1, min(int(min_anchor_questions), max_inf))
        probes = max(0, min(int(max_generated_probes), max_inf - anchors))
        return cls(
            max_inference_questions=max_inf,
            min_anchor_questions=anchors,
            max_generated_probes=probes,
        )


@dataclass(frozen=True)
class ScoredQuestion:
    question: Question
    score: float
    components: Dict[str, float] = field(default_factory=dict)


def _source(question_id: str, generated_ids: Set[str]) -> str:
    return "generated" if question_id in generated_ids else "seed"


def _calibration_status(question_id: str, question_metadata: Dict[str, Dict[str, Any]]) -> str:
    meta = question_metadata.get(question_id, {})
    return str(meta.get("calibration_status") or ("accepted_uncalibrated" if meta.get("source") == "generated" else "calibrated"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _novelty_by_question(questions: Sequence[Question], asked_questions: Sequence[Question]) -> Dict[str, float]:
    if not questions:
        return {}
    if not asked_questions:
        return {q.id: 1.0 for q in questions}

    texts = [q.text for q in questions]
    asked_texts = [q.text for q in asked_questions]
    emb = embed_texts(texts + asked_texts)
    q_emb = emb[: len(texts)]
    asked_emb = emb[len(texts) :]
    sims = cosine_similarity_matrix(q_emb, asked_emb)
    max_sims = np.max(sims, axis=1)
    return {
        q.id: float(max(0.0, min(1.0, 1.0 - max_sims[i])))
        for i, q in enumerate(questions)
    }


def select_next_question_exploratory(
    state: "PersonalityState",
    questions: List[Question],
    *,
    asked_ids: Optional[Set[str]] = None,
    generated_question_ids: Optional[Set[str]] = None,
    question_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    composition: Optional[SessionComposition] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[Question], float, List[Tuple[Question, float, Dict[str, float]]]]:
    """
    Select a question with anchored exploratory scoring.

    Hard rules are applied before scoring:
    - ask seed/anchor questions until min_anchor_questions is reached.
    - stop selecting generated probes after max_generated_probes have been asked.
    """
    from models.personality_state import PersonalityState

    if not isinstance(state, PersonalityState):
        raise TypeError("state must be a PersonalityState instance.")

    asked = asked_ids or set()
    generated_ids = generated_question_ids or set()
    metadata = question_metadata or {}
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    comp = composition or SessionComposition.scaled(len(questions))

    id_to_q = {q.id: q for q in questions}
    asked_seed = sum(1 for qid in asked if _source(qid, generated_ids) == "seed")
    asked_generated = sum(1 for qid in asked if _source(qid, generated_ids) == "generated")

    candidates = [q for q in questions if q.id not in asked]
    if asked_seed < comp.min_anchor_questions:
        candidates = [q for q in candidates if _source(q.id, generated_ids) == "seed"]
    if asked_generated >= comp.max_generated_probes:
        candidates = [q for q in candidates if _source(q.id, generated_ids) != "generated"]

    if not candidates:
        return None, 0.0, []

    asked_questions = [id_to_q[qid] for qid in asked if qid in id_to_q]
    novelty = _novelty_by_question(candidates, asked_questions)

    scored: List[ScoredQuestion] = []
    for q in candidates:
        src = _source(q.id, generated_ids)
        eig = float(expected_information_gain(state, q))
        nov = float(novelty.get(q.id, 1.0))
        redundancy = 1.0 - nov
        status = _calibration_status(q.id, metadata)
        exploration = 1.0 if src == "generated" and status in {"candidate", "accepted_uncalibrated", "in_exploration"} else 0.0
        policy_prior = _safe_float(metadata.get(q.id, {}).get("policy_quality_prior"), 0.0)
        risk_penalty = _safe_float(metadata.get(q.id, {}).get("risk_penalty"), 0.0)
        score = (
            w["eig"] * eig
            + w["novelty"] * nov
            + w["exploration"] * exploration
            + w["policy"] * policy_prior
            - w["redundancy"] * redundancy
            - w["risk"] * risk_penalty
        )
        scored.append(
            ScoredQuestion(
                question=q,
                score=float(score),
                components={
                    "selection_score": float(score),
                    "expected_information_gain": eig,
                    "semantic_novelty": nov,
                    "exploration_bonus": float(exploration),
                    "policy_quality_prior": float(policy_prior),
                    "redundancy_penalty": float(redundancy),
                    "risk_penalty": float(risk_penalty),
                },
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    return (
        scored[0].question,
        scored[0].score,
        [(item.question, item.score, item.components) for item in scored],
    )
