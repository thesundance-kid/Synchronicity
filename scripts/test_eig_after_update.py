#!/usr/bin/env python3
"""
Sanity test: EIG ranking before vs after one strong extraversion response.

Run from project root: python scripts/test_eig_after_update.py

Verifies that after updating the belief with a strong (Likert=5) response to
question e_01, the EIG ranking is no longer fully symmetric and the selector
still works when excluding e_01 via asked_ids.
"""

import os
import sys

# Project root = parent of scripts/
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.personality_state import PersonalityState
from models.question_bank import load_questions
from models.question_selection import select_next_question_eig


def main() -> None:
    questions_path = os.path.join(_project_root, "data", "questions.json")
    if not os.path.isfile(questions_path):
        print("Error: data/questions.json not found at", questions_path)
        sys.exit(1)

    questions = load_questions(questions_path, expected_dim=5)
    e_01 = next((q for q in questions if q.id == "e_01"), None)
    if e_01 is None:
        print("Error: question e_01 not found in question bank")
        sys.exit(1)

    # Neutral 5D state (existing API: dim=5 => mu=0, sigma=4*I)
    state = PersonalityState(dim=5)

    # ---------- BEFORE UPDATE ----------
    print("=" * 60)
    print("BEFORE UPDATE (neutral prior, no responses yet)")
    print("=" * 60)

    best_before, score_before, ranked_before = select_next_question_eig(state, questions)
    print("\nTop 10 EIG-ranked questions (before any update):")
    for i, (q, s) in enumerate(ranked_before[:10], 1):
        print("  {:2}. {}  EIG = {:.8f}".format(i, q.id, s))

    print("\nBEST NEXT QUESTION (before): {}  (EIG = {:.8f})".format(best_before.id, score_before))

    # Apply strong positive response to extraversion question e_01
    state.update_posterior_likert_laplace(
        w=e_01.w,
        y=5,
        thresholds=e_01.thresholds,
        noise_var=e_01.noise_var,
    )

    # ---------- State after update ----------
    print("\n" + "=" * 60)
    print("State after one strong response (y=5) to e_01")
    print("=" * 60)
    print("Updated mu (trait means):", state.mu)
    print("Updated variances (diag of sigma):", state.variances())

    # ---------- AFTER UPDATE ----------
    print("\n" + "=" * 60)
    print("AFTER UPDATE (excluding e_01 via asked_ids)")
    print("=" * 60)

    best_after, score_after, ranked_after = select_next_question_eig(
        state, questions, asked_ids={"e_01"}
    )
    print("\nTop 10 EIG-ranked questions (after update, e_01 excluded):")
    for i, (q, s) in enumerate(ranked_after[:10], 1):
        print("  {:2}. {}  EIG = {:.8f}".format(i, q.id, s))

    # Position and EIG for remaining extraversion questions (e_02, e_03, e_04)
    id_to_rank_score = {q.id: (rank, s) for rank, (q, s) in enumerate(ranked_after, 1)}
    print("\nRemaining extraversion questions (after update):")
    for eid in ("e_02", "e_03", "e_04"):
        if eid in id_to_rank_score:
            pos, eig = id_to_rank_score[eid]
            print("  {}  rank = {:2},  EIG = {:.8f}".format(eid, pos, eig))
        else:
            print("  {}  (not in ranking)".format(eid))

    print("\nBottom 10 ranked questions (after update):")
    n = len(ranked_after)
    for i, (q, s) in enumerate(ranked_after[-10:], start=n - 9):
        print("  {:2}. {}  EIG = {:.8f}".format(i, q.id, s))

    print("\nBEST NEXT QUESTION (after): {}  (EIG = {:.8f})".format(best_after.id, score_after))

    # ---------- Interpretation ----------
    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)
    print("- Before update: all questions are effectively tied because of symmetry")
    print("  (neutral prior and identical question structure per trait).")
    print("- After update: the already-probed trait axis (extraversion) should become")
    print("  less informative, so e_02, e_03, e_04 tend to receive lower EIG and rank")
    print("  lower. The untouched traits (O, C, A, N) remain roughly tied and more")
    print("  informative to ask next.")
    print()


if __name__ == "__main__":
    main()
