"""
Adaptive question selection via Expected Information Gain (EIG).

EIG(q) = H(current state) - E_y[ H(posterior after observing y) ]

We score each candidate question by how much entropy we expect to reduce
on average after asking it, then select the question with the highest EIG.
Hypothetical updates are performed on a copy of the state so the current
belief is never mutated.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np

# Type alias for state: any object with copy(), entropy(), predict_likert_probs(), update_posterior_likert_laplace().
# We depend only on the PersonalityState interface from personality_state.py.
from models.question_bank import Question, load_questions


def expected_information_gain(
    state: "PersonalityState",
    question: Question,
    min_prob: float = 1e-4,
) -> float:
    """
    Compute Expected Information Gain for asking this question.

    EIG(q) = H(current) - E_y[ H(posterior | y) ]
    where the expectation is over the predictive distribution P(y) under the
    current state. We only include response categories y with P(y) >= min_prob
    to avoid expensive Laplace updates for negligible probability mass.

    Args:
        state: Current Gaussian belief (PersonalityState). Not mutated.
        question: Candidate question with w, thresholds, noise_var.
        min_prob: Skip hypothetical response y if P(y) < min_prob.

    Returns:
        EIG in nats (non-negative in theory; can be slightly negative numerically).

    Math Note:
        Information gain for a single outcome y is:
            IG(y) = H(prior) - H(posterior after y).
        EIG is the average over outcomes weighted by P(y):
            EIG = sum_y P(y) * IG(y) = H(prior) - sum_y P(y) * H(posterior|y).
    """
    # Avoid circular import at module load; only need type for docstring.
    from models.personality_state import PersonalityState

    if not isinstance(state, PersonalityState):
        raise TypeError("state must be a PersonalityState instance.")

    H_current = state.entropy()

    # Predictive distribution over Likert categories 1..K.
    probs = state.predict_likert_probs(
        w=question.w,
        thresholds=question.thresholds,
        noise_var=question.noise_var,
    )
    K = probs.size

    expected_H_posterior = 0.0
    total_mass = 0.0
    for y in range(1, K + 1):
        p = probs[y - 1]
        if p < min_prob:
            continue
        # Simulate posterior after observing y on a COPY so we never mutate state.
        state_copy = state.copy()
        state_copy.update_posterior_likert_laplace(
            w=question.w,
            y=y,
            thresholds=question.thresholds,
            noise_var=question.noise_var,
        )
        expected_H_posterior += p * state_copy.entropy()
        total_mass += p

    # If no branch met min_prob, return 0 to avoid meaningless EIG.
    if total_mass <= 0:
        return 0.0

    # Do NOT renormalize by total_mass. The correct formula is:
    #   EIG = H_current - sum_y p_y * H(posterior | y)
    # Skipping low-probability branches (p < min_prob) is a numerical
    # approximation, but dividing by total_mass would compute the *conditional*
    # expected entropy given the response falls in the included set — a
    # different quantity that biases EIG downward for questions with any
    # probability mass in skipped categories.
    return float(H_current - expected_H_posterior)


def select_next_question_eig(
    state: "PersonalityState",
    questions: List[Question],
    asked_ids: Optional[Set[str]] = None,
    min_prob: float = 1e-4,
) -> Tuple[Optional[Question], float, List[Tuple[Question, float]]]:
    """
    Select the next question to ask by maximizing Expected Information Gain.

    Excludes any question whose id is in asked_ids. Scores all remaining
    questions and returns the best one plus a full ranking.

    Args:
        state: Current Gaussian belief. Not mutated.
        questions: Full list of candidate questions.
        asked_ids: Set of question ids already asked; these are excluded.
            If None, no questions are excluded.
        min_prob: Passed through to expected_information_gain.

    Returns:
        (best_question, best_score, ranked_results)
        - best_question: The question with highest EIG, or None if no candidates.
        - best_score: EIG of the best question (0.0 if no candidates).
        - ranked_results: List of (question, score) sorted descending by score.
    """
    from models.personality_state import PersonalityState

    if not isinstance(state, PersonalityState):
        raise TypeError("state must be a PersonalityState instance.")
    asked = asked_ids if asked_ids is not None else set()

    candidates = [q for q in questions if q.id not in asked]
    if not candidates:
        return None, 0.0, []

    scored: List[Tuple[Question, float]] = []
    for q in candidates:
        eig = expected_information_gain(state, q, min_prob=min_prob)
        scored.append((q, eig))

    # Sort descending by EIG (higher = more informative).
    scored.sort(key=lambda x: x[1], reverse=True)
    best_question, best_score = scored[0]

    return best_question, best_score, scored


# ---------------------------------------------------------------------------
# Smoke test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    from models.personality_state import PersonalityState

    # Resolve path to data/questions.json relative to project root (parent of models/).
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    questions_path = os.path.join(project_root, "data", "questions.json")

    print("=== Adaptive question selection demo ===\n")

    # Neutral 5D state (matches question dimension).
    state = PersonalityState(dim=5)
    print("State: dim=5, neutral prior (mu=0, sigma=4*I)")
    print("Current entropy: {:.4f} nats\n".format(state.entropy()))

    questions = load_questions(questions_path, expected_dim=5)
    print("Loaded {} questions from {}\n".format(len(questions), questions_path))

    # EIG for first few questions.
    print("EIG for first 5 questions:")
    for q in questions[:5]:
        eig = expected_information_gain(state, q)
        print("  {} ({}): EIG = {:.4f}".format(q.id, q.text[:40] + "...", eig))

    # Select next question and show top ranked.
    best, score, ranked = select_next_question_eig(state, questions)
    print("\nBest next question: {} (EIG = {:.4f})".format(best.id, score))
    print("Top 5 ranked by EIG:")
    for q, s in ranked[:5]:
        print("  {} : {:.4f}".format(q.id, s))

    # Verify asked_ids exclusion: after "asking" one question, it should not be selected.
    asked_ids = {best.id}
    best2, score2, ranked2 = select_next_question_eig(state, questions, asked_ids=asked_ids)
    print("\nAfter marking '{}' as asked:".format(best.id))
    print("  Next best: {} (EIG = {:.4f})".format(best2.id if best2 else "None", score2))
    assert best2 is not None and best2.id != best.id, "asked_ids should exclude previous best."
    print("  asked_ids exclusion verified.")
