"""
Phase 1 smoke tests: anonymous users, warm-start, posterior snapshots.

Covers:
- anonymous user registration
- cold prior (new user starts from N(0, I))
- posterior snapshots: step_idx=0 plus one per inference answer
- user_current_state updated after every inference answer
- warm-start: new session loads prior from user_current_state, not N(0, I)
- user_posteriors written only when a session reaches complete
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db
from app.session_manager import QuestionPayload, create_session, get_next_question, record_answer

QUESTIONS_PATH = str(PROJECT_ROOT / "data" / "questions_v2.json")
TEST_DB_PATH = str(PROJECT_ROOT / "data" / "pilot_phase1_test.db")

_PASS = 0
_FAIL = 0


def _ok(name: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  PASS  {name}")


def _fail(name: str, msg: str) -> None:
    global _FAIL
    _FAIL += 1
    print(f"  FAIL  {name}: {msg}")


def _assert(cond: bool, name: str, msg: str = "") -> None:
    if cond:
        _ok(name)
    else:
        _fail(name, msg or "assertion failed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_conn() -> db.sqlite3.Connection:
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    conn = db.connect(TEST_DB_PATH)
    db.init_db(conn)
    return conn


def new_user(conn) -> str:
    uid = secrets.token_urlsafe(16)
    db.insert_user(conn, user_id=uid)
    return uid


def run_inference(conn, session_id: str, first_q: QuestionPayload, n: int, response: int = 3) -> int:
    """Answer up to n inference questions; return how many were answered."""
    q: Optional[QuestionPayload] = first_q
    answered = 0
    while q is not None and q.pool == "inference" and answered < n:
        out = record_answer(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            session_id=session_id,
            question_id=q.id,
            response=response,
            dim=5,
        )
        answered += 1
        nq = out["next_question"]
        q = QuestionPayload(**nq) if nq is not None else None
    return answered


def complete_session(conn, session_id: str, first_q: QuestionPayload) -> None:
    """Drive a session to completion answering all inference then heldout questions."""
    q: Optional[QuestionPayload] = first_q
    while q is not None:
        out = record_answer(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            session_id=session_id,
            question_id=q.id,
            response=3,
            dim=5,
        )
        nq = out["next_question"]
        q = QuestionPayload(**nq) if nq is not None else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_register_user(conn) -> str:
    uid = new_user(conn)
    user = db.get_user(conn, uid)
    _assert(user is not None, "register_user/get_user returns row")
    _assert(user["user_id"] == uid, "register_user/user_id matches")
    return uid


def test_cold_prior(conn, user_id: str):
    """New user gets flat N(0, I) prior at step_idx=0."""
    session_id, first_q = create_session(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        mode="adaptive",
        max_inference_questions=3,
        num_heldout=2,
        dim=5,
        user_id=user_id,
    )
    snaps = db.list_posterior_snapshots(conn, session_id)
    _assert(len(snaps) >= 1, "cold_prior/snapshot_exists")
    _assert(snaps[0]["step_idx"] == 0, "cold_prior/step_idx_is_0")
    mu = np.array(snaps[0]["mu"])
    _assert(np.allclose(mu, 0.0, atol=1e-6), "cold_prior/mu_is_zero", f"mu={mu}")
    # No user_current_state yet (no inference answers given)
    state = db.get_user_current_state(conn, user_id)
    _assert(state is None, "cold_prior/no_current_state_before_answers")
    return session_id, first_q


def test_posterior_snapshots(conn, session_id: str, first_q: QuestionPayload):
    """Step_idx=0 plus one snapshot per inference answer."""
    n_inference = 3
    answered = run_inference(conn, session_id, first_q, n=n_inference, response=5)
    _assert(answered == n_inference, "snapshots/all_answered", f"answered={answered}")

    snaps = db.list_posterior_snapshots(conn, session_id)
    expected = n_inference + 1  # step 0 + 3 updates
    _assert(len(snaps) == expected, "snapshots/count", f"got {len(snaps)}, want {expected}")

    step_indices = [s["step_idx"] for s in snaps]
    _assert(
        step_indices == list(range(expected)),
        "snapshots/step_indices",
        f"got {step_indices}",
    )

    # Entropy should not increase significantly over inference (Laplace approx may fluctuate
    # slightly, so allow a small tolerance).
    first_entropy = snaps[0]["entropy"]
    last_entropy = snaps[-1]["entropy"]
    _assert(
        first_entropy >= last_entropy - 0.5,
        "snapshots/entropy_non_increasing",
        f"first={first_entropy:.4f}, last={last_entropy:.4f}",
    )


def test_user_current_state(conn, user_id: str, session_id: str):
    """user_current_state reflects the latest inference step."""
    state = db.get_user_current_state(conn, user_id)
    _assert(state is not None, "current_state/exists")
    _assert(state["latest_session_id"] == session_id, "current_state/session_id")
    _assert(state["latest_step_idx"] > 0, "current_state/step_idx_positive",
            f"step_idx={state['latest_step_idx']}")
    mu = np.array(state["mu"])
    _assert(mu.shape == (5,), "current_state/mu_shape", f"shape={mu.shape}")
    # With response=5 (strongly agree) and openness/extraversion items, mu should move away from 0.
    _assert(not np.allclose(mu, 0.0, atol=1e-6), "current_state/mu_not_zero",
            "posterior mu stayed at zero after responses")


def test_warm_start(conn, user_id: str):
    """New session for returning user warm-starts from user_current_state."""
    prior_state = db.get_user_current_state(conn, user_id)
    _assert(prior_state is not None, "warmstart/prior_state_available")

    session_id2, _ = create_session(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        mode="adaptive",
        max_inference_questions=3,
        num_heldout=2,
        dim=5,
        user_id=user_id,
    )

    snaps2 = db.list_posterior_snapshots(conn, session_id2)
    _assert(len(snaps2) >= 1, "warmstart/snapshot_exists")
    _assert(snaps2[0]["step_idx"] == 0, "warmstart/step_idx_0")

    mu_warm = np.array(snaps2[0]["mu"])
    mu_prior = np.array(prior_state["mu"])
    _assert(
        np.allclose(mu_warm, mu_prior, atol=1e-9),
        "warmstart/mu_matches_prior_state",
        f"\n  expected: {mu_prior}\n  got:      {mu_warm}",
    )

    # prior_session_id should point to the earlier session.
    sess2 = db.get_session(conn, session_id2)
    _assert(
        sess2.prior_session_id == prior_state["latest_session_id"],
        "warmstart/prior_session_id",
        f"expected {prior_state['latest_session_id']}, got {sess2.prior_session_id}",
    )
    return session_id2


def test_user_posteriors_on_completion(conn, user_id: str):
    """user_posteriors is written exactly once when a session completes."""
    before = db.count_user_posteriors(conn, user_id)

    session_id, first_q = create_session(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        mode="adaptive",
        max_inference_questions=2,
        num_heldout=2,
        dim=5,
        user_id=user_id,
    )
    complete_session(conn, session_id, first_q)

    sess = db.get_session(conn, session_id)
    _assert(sess.status == "complete", "user_posteriors/session_complete",
            f"status={sess.status}")

    after = db.count_user_posteriors(conn, user_id)
    _assert(after == before + 1, "user_posteriors/count_incremented",
            f"before={before}, after={after}")

    latest = db.get_latest_user_posterior(conn, user_id)
    _assert(latest is not None, "user_posteriors/row_exists")
    _assert(latest["session_id"] == session_id, "user_posteriors/session_id")
    _assert(latest["session_number"] == before + 1, "user_posteriors/session_number",
            f"got {latest['session_number']}")
    mu = np.array(latest["mu"])
    _assert(mu.shape == (5,), "user_posteriors/mu_shape", f"shape={mu.shape}")


def test_anonymous_session_unaffected(conn):
    """Sessions without user_id continue to work: no snapshot write crash, no user_current_state."""
    session_id, first_q = create_session(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        mode="adaptive",
        max_inference_questions=2,
        num_heldout=2,
        dim=5,
        # no user_id
    )
    # Snapshot at step 0 should still be written.
    snaps = db.list_posterior_snapshots(conn, session_id)
    _assert(len(snaps) == 1 and snaps[0]["step_idx"] == 0,
            "anon/step0_snapshot_written")

    # Answer one inference question.
    out = record_answer(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        session_id=session_id,
        question_id=first_q.id,
        response=3,
        dim=5,
    )
    snaps = db.list_posterior_snapshots(conn, session_id)
    _assert(len(snaps) == 2, "anon/step1_snapshot_written", f"got {len(snaps)}")

    sess = db.get_session(conn, session_id)
    _assert(sess.user_id is None, "anon/no_user_id")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Phase 1 User Tests ===\n")
    conn = fresh_conn()
    try:
        user_id = test_register_user(conn)
        session_id1, first_q1 = test_cold_prior(conn, user_id)
        test_posterior_snapshots(conn, session_id1, first_q1)
        test_user_current_state(conn, user_id, session_id1)
        test_warm_start(conn, user_id)
        test_user_posteriors_on_completion(conn, user_id)
        test_anonymous_session_unaffected(conn)
    finally:
        conn.close()
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    print(f"\n{'='*40}")
    total = _PASS + _FAIL
    print(f"Results: {_PASS}/{total} passed, {_FAIL} failed")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
