"""
Held-out evaluation utilities for real-user pilot sessions.

These metrics are computed from:
- a completed posterior (PersonalityState)
- held-out questions + their ground-truth responses
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from models.personality_state import PersonalityState
from models.question_bank import Question


@dataclass(frozen=True)
class HeldoutMetrics:
    n: int
    heldout_log_likelihood: float
    mean_true_prob: float
    accuracy: Optional[float]


def evaluate_heldout(
    *,
    state: PersonalityState,
    heldout_questions: Sequence[Question],
    heldout_responses: Mapping[str, int],
    compute_accuracy: bool = True,
) -> HeldoutMetrics:
    """
    Evaluate held-out predictive performance.

    Args:
        state: Posterior after inference questions.
        heldout_questions: Held-out items (Question objects).
        heldout_responses: Map from question_id -> observed response (1..K).
        compute_accuracy: If True, compute exact-category accuracy using argmax.

    Returns:
        HeldoutMetrics
    """
    if not heldout_questions:
        return HeldoutMetrics(n=0, heldout_log_likelihood=0.0, mean_true_prob=0.0, accuracy=None)

    loglik = 0.0
    true_probs: list[float] = []
    correct = 0
    used = 0

    for q in heldout_questions:
        if q.id not in heldout_responses:
            continue
        y = int(heldout_responses[q.id])
        probs = state.predict_likert_probs(w=q.w, thresholds=q.thresholds, noise_var=float(q.noise_var))
        K = int(probs.size)
        if y < 1 or y > K:
            raise ValueError(f"Held-out response for {q.id} must be in [1, {K}], got {y}")
        p_true = float(probs[y - 1])
        p_true = max(p_true, 1e-12)
        loglik += math.log(p_true)
        true_probs.append(p_true)

        if compute_accuracy:
            y_hat = int(np.argmax(probs) + 1)
            if y_hat == y:
                correct += 1
        used += 1

    if used == 0:
        return HeldoutMetrics(n=0, heldout_log_likelihood=0.0, mean_true_prob=0.0, accuracy=None)

    acc = (correct / used) if compute_accuracy else None
    return HeldoutMetrics(
        n=used,
        heldout_log_likelihood=float(loglik),
        mean_true_prob=float(sum(true_probs) / used),
        accuracy=float(acc) if acc is not None else None,
    )


def evaluate_heldout_performance(
    posterior: PersonalityState,
    heldout_items: Sequence[Question],
    responses: Mapping[str, int],
) -> Dict[str, Any]:
    """
    Summarize held-out predictive quality using the posterior **after inference only**
    (no updates from held-out responses).

    Returns:
        {
            "heldout_log_likelihood": float,
            "mean_true_prob": float,
            "n_questions": int,
        }
    """
    m = evaluate_heldout(
        state=posterior,
        heldout_questions=heldout_items,
        heldout_responses=responses,
        compute_accuracy=False,
    )
    return {
        "heldout_log_likelihood": float(m.heldout_log_likelihood),
        "mean_true_prob": float(m.mean_true_prob),
        "n_questions": int(m.n),
    }

