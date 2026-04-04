"""
Local smoke test for the real-user pilot backend (no web server required).

It exercises:
- creating a session (adaptive mode)
- recording a few inference answers (posterior updates)
- transitioning to held-out questions
- computing held-out evaluation metrics
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db
from app.session_manager import create_session, get_next_question, get_session_summary, record_answer
from models.personality_state import PersonalityState
from models.question_bank import load_question_pools_v2
from models.real_eval import evaluate_heldout


def main() -> None:
    questions_path = str(PROJECT_ROOT / "data" / "questions_v2.json")
    test_db_path = str(PROJECT_ROOT / "data" / "pilot_test.db")

    # Reset test DB for a clean run.
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    conn = db.connect(test_db_path)
    db.init_db(conn)

    session_id, first_q = create_session(
        conn,
        questions_v2_path=questions_path,
        mode="adaptive",
        max_inference_questions=3,
        num_heldout=2,
        dim=5,
    )
    print("Session created:", session_id)
    print("First question:", first_q)

    # Hard-coded answers for reproducibility.
    # (1..5 Likert categories)
    scripted_answers = [5, 3, 1]

    q = first_q
    for ans in scripted_answers:
        out = record_answer(
            conn,
            questions_v2_path=questions_path,
            session_id=session_id,
            question_id=q.id,
            response=ans,
            dim=5,
        )
        print("\nAnswered:", q.id, "=", ans)
        print("State:", {"status": out["status"], "step": out["step"]})
        if out["next_question"] is None:
            break
        q = type(first_q)(**out["next_question"])  # QuestionPayload-like
        print("Next question:", q.id, q.pool)

    # We should now be in heldout (given max_inference_questions=3).
    summary = get_session_summary(conn, questions_v2_path=questions_path, session_id=session_id, dim=5)
    print("\nSession summary (truncated):")
    print(
        {
            "session_id": summary["session_id"],
            "mode": summary["mode"],
            "status": summary["status"],
            "asked_question_ids": summary["asked_question_ids"],
            "heldout_question_ids": summary["heldout_question_ids"],
            "num_responses": len(summary["responses"]),
        }
    )

    # Answer held-out questions with dummy values.
    # (In a real pilot, you would collect these from the participant.)
    while True:
        nq = get_next_question(conn, questions_v2_path=questions_path, session_id=session_id, dim=5)
        if nq is None:
            break
        if nq.pool != "heldout":
            raise RuntimeError(f"Expected heldout question, got {nq.pool}")
        out = record_answer(
            conn,
            questions_v2_path=questions_path,
            session_id=session_id,
            question_id=nq.id,
            response=4,
            dim=5,
        )
        print("Heldout answered:", nq.id, "=", 4, "->", out["status"])
        if out["status"] == "complete":
            break

    # Refresh summary after held-out answers.
    summary = get_session_summary(conn, questions_v2_path=questions_path, session_id=session_id, dim=5)

    # Evaluate held-out metrics from the final posterior snapshot.
    inference_pool, heldout_pool = load_question_pools_v2(questions_path, expected_dim=5)
    sess = db.get_session(conn, session_id)
    state = PersonalityState(mu_init=np.array(sess.posterior_mu), sigma_init=np.array(sess.posterior_sigma))

    heldout_ids = set(sess.heldout_ids)
    heldout_questions = [q for q in heldout_pool if q.id in heldout_ids]
    heldout_responses = {
        r["question_id"]: int(r["response"])
        for r in summary["responses"]
        if r["pool"] == "heldout"
    }
    metrics = evaluate_heldout(
        state=state,
        heldout_questions=heldout_questions,
        heldout_responses=heldout_responses,
        compute_accuracy=True,
    )
    print("\nHeld-out metrics:", metrics)

    conn.close()
    print("\nOK")


if __name__ == "__main__":
    main()

