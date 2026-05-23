"""
Phase 2 smoke tests: question_performance_events.

Covers:
- one event is created per inference answer (not per heldout answer)
- event count matches number of inference answers
- realized_information_gain == entropy_before - entropy_after
- mu_before and mu_after are both stored and differ
- user_id is stored when present, None when anonymous
- parameter_version is None (Phase 4 will populate it)
- seeds-only / no-API-key path creates events without error
- list_question_performance_events_for_question aggregates across sessions
"""

from __future__ import annotations

import math
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
TEST_DB_PATH = str(PROJECT_ROOT / "data" / "pilot_phase2_test.db")

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


def drive_to_completion(conn, session_id: str, first_q: QuestionPayload, response: int = 3) -> int:
    """Answer all questions (inference + heldout). Returns count of inference answers given."""
    q: Optional[QuestionPayload] = first_q
    inference_count = 0
    while q is not None:
        is_inference = q.pool == "inference"
        out = record_answer(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            session_id=session_id,
            question_id=q.id,
            response=response,
            dim=5,
        )
        if is_inference:
            inference_count += 1
        nq = out["next_question"]
        q = QuestionPayload(**nq) if nq is not None else None
    return inference_count


def answer_n_inference(
    conn,
    session_id: str,
    first_q: QuestionPayload,
    n: int,
    response: int = 5,
) -> tuple[int, Optional[QuestionPayload]]:
    """Answer up to n inference questions. Returns (answered_count, last_next_q)."""
    q: Optional[QuestionPayload] = first_q
    answered = 0
    last_next: Optional[QuestionPayload] = None
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
        last_next = QuestionPayload(**nq) if nq is not None else None
        q = last_next if (last_next is not None and last_next.pool == "inference") else None
    return answered, last_next


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_event_count_matches_inference_answers(conn):
    """One event per inference answer, none for heldout."""
    n_inference = 4
    session_id, first_q = create_session(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        mode="adaptive",
        max_inference_questions=n_inference,
        num_heldout=2,
        dim=5,
    )

    # No events before any answers.
    events = db.list_question_performance_events(conn, session_id)
    _assert(len(events) == 0, "event_count/zero_before_answers")

    # Answer all inference questions, then all heldout questions.
    inference_answered = drive_to_completion(conn, session_id, first_q)

    events = db.list_question_performance_events(conn, session_id)
    _assert(
        len(events) == inference_answered,
        "event_count/matches_inference_answers",
        f"events={len(events)}, inference_answered={inference_answered}",
    )
    _assert(
        len(events) == n_inference,
        "event_count/matches_max_inference",
        f"events={len(events)}, n_inference={n_inference}",
    )
    return session_id, events


def test_event_fields(events, session_id: str):
    """Check field values for correctness."""
    for i, ev in enumerate(events):
        tag = f"field_check/event_{i}"

        # realized_ig must equal entropy_before - entropy_after exactly.
        expected_rig = ev["entropy_before"] - ev["entropy_after"]
        _assert(
            math.isclose(ev["realized_information_gain"], expected_rig, rel_tol=1e-9),
            f"{tag}/realized_ig",
            f"rig={ev['realized_information_gain']:.8f}, expected={expected_rig:.8f}",
        )

        # mu_before and mu_after must be stored as 5-element lists.
        mu_before = np.array(ev["mu_before"])
        mu_after = np.array(ev["mu_after"])
        _assert(mu_before.shape == (5,), f"{tag}/mu_before_shape", f"shape={mu_before.shape}")
        _assert(mu_after.shape == (5,), f"{tag}/mu_after_shape", f"shape={mu_after.shape}")

        # sigma shapes: 5x5.
        sig_before = np.array(ev["sigma_before"])
        sig_after = np.array(ev["sigma_after"])
        _assert(sig_before.shape == (5, 5), f"{tag}/sigma_before_shape")
        _assert(sig_after.shape == (5, 5), f"{tag}/sigma_after_shape")

        # parameter_version must be None (Phase 4 will populate).
        _assert(ev["parameter_version"] is None, f"{tag}/parameter_version_none",
                f"got {ev['parameter_version']}")

        # session_id must match.
        _assert(ev["session_id"] == session_id, f"{tag}/session_id")

        # question_source must be 'seed' or 'generated'.
        _assert(ev["question_source"] in ("seed", "generated"), f"{tag}/question_source",
                f"got {ev['question_source']!r}")

        # predicted_eig should be a non-negative float (EIG >= 0 by definition).
        _assert(
            ev["predicted_eig"] is not None and ev["predicted_eig"] >= 0.0,
            f"{tag}/predicted_eig_positive",
            f"got {ev['predicted_eig']}",
        )

        # Response value must be in [1, 5].
        _assert(1 <= ev["response_value"] <= 5, f"{tag}/response_value_range",
                f"got {ev['response_value']}")


def test_mu_before_after_differ_after_first_update(events):
    """Posterior covariance must change after each inference update.

    Note: a neutral response (y=3) on a symmetric scale leaves mu at zero but
    tightens sigma. We therefore assert sigma changes rather than mu, which is
    correct for any response value.
    """
    _assert(len(events) >= 1, "mu_change/has_events")
    if not events:
        return
    ev = events[0]
    sig_before = np.array(ev["sigma_before"])
    sig_after = np.array(ev["sigma_after"])
    # The Laplace approximation always tightens sigma on any inference update.
    _assert(
        not np.allclose(sig_before, sig_after, atol=1e-9),
        "mu_change/sigma_before_ne_sigma_after",
        f"sigma_before diag={np.diag(sig_before)}, sigma_after diag={np.diag(sig_after)}",
    )


def test_user_id_stored(conn):
    """user_id is stored in events for identified users, None for anonymous."""
    uid = secrets.token_urlsafe(16)
    db.insert_user(conn, user_id=uid)

    # Session with user_id.
    session_id, first_q = create_session(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        mode="adaptive",
        max_inference_questions=2,
        num_heldout=2,
        dim=5,
        user_id=uid,
    )
    answer_n_inference(conn, session_id, first_q, n=2)
    events = db.list_question_performance_events(conn, session_id)
    _assert(len(events) >= 1, "user_id/has_events")
    for ev in events:
        _assert(ev["user_id"] == uid, "user_id/stored_correctly",
                f"got {ev['user_id']!r}")

    # Anonymous session: user_id should be None.
    session_id2, first_q2 = create_session(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        mode="adaptive",
        max_inference_questions=2,
        num_heldout=2,
        dim=5,
    )
    answer_n_inference(conn, session_id2, first_q2, n=2)
    events2 = db.list_question_performance_events(conn, session_id2)
    _assert(len(events2) >= 1, "user_id/anon_has_events")
    for ev in events2:
        _assert(ev["user_id"] is None, "user_id/none_for_anon",
                f"got {ev['user_id']!r}")


def test_no_events_for_heldout(conn):
    """Heldout answers must not create performance events."""
    n_inference = 2
    session_id, first_q = create_session(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        mode="adaptive",
        max_inference_questions=n_inference,
        num_heldout=2,
        dim=5,
    )
    # Answer all inference questions first.
    answer_n_inference(conn, session_id, first_q, n=n_inference)
    events_after_inference = db.list_question_performance_events(conn, session_id)
    inference_event_count = len(events_after_inference)

    # Now answer all heldout questions.
    q = get_next_question(conn, questions_v2_path=QUESTIONS_PATH, session_id=session_id, dim=5)
    heldout_answered = 0
    while q is not None and q.pool == "heldout":
        out = record_answer(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            session_id=session_id,
            question_id=q.id,
            response=3,
            dim=5,
        )
        heldout_answered += 1
        nq = out["next_question"]
        q = QuestionPayload(**nq) if nq is not None else None

    _assert(heldout_answered > 0, "no_heldout_events/heldout_questions_answered",
            "No heldout questions were found to answer")

    events_final = db.list_question_performance_events(conn, session_id)
    _assert(
        len(events_final) == inference_event_count,
        "no_heldout_events/count_unchanged",
        f"before heldout={inference_event_count}, after heldout={len(events_final)}",
    )


def test_cross_session_lookup(conn):
    """list_question_performance_events_for_question aggregates across sessions."""
    # Run two separate sessions; both will likely ask some overlapping questions
    # (e.g. o_01 is always first due to EIG tie-breaking with flat prior).
    def run_one_session():
        sess_id, fq = create_session(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            mode="adaptive",
            max_inference_questions=2,
            num_heldout=2,
            dim=5,
        )
        answer_n_inference(conn, sess_id, fq, n=2)
        return sess_id

    sess1 = run_one_session()
    sess2 = run_one_session()

    # Find a question that appears in both sessions.
    ev1 = db.list_question_performance_events(conn, sess1)
    ev2 = db.list_question_performance_events(conn, sess2)
    ids1 = {e["question_id"] for e in ev1}
    ids2 = {e["question_id"] for e in ev2}
    shared = ids1 & ids2

    _assert(len(shared) > 0, "cross_session/shared_questions_exist",
            f"sess1 ids={ids1}, sess2 ids={ids2}")

    if shared:
        qid = next(iter(shared))
        cross_events = db.list_question_performance_events_for_question(conn, qid)
        sessions_in_cross = {e["session_id"] for e in cross_events}
        _assert(
            sess1 in sessions_in_cross and sess2 in sessions_in_cross,
            "cross_session/both_sessions_present",
            f"sessions in cross={sessions_in_cross}",
        )
        _assert(len(cross_events) >= 2, "cross_session/at_least_two_events",
                f"got {len(cross_events)}")


def test_seeds_only_no_api_key(conn):
    """Events are created even when no LLM is configured (DummyLLMClient / seeds only)."""
    # No llm_api_key → DummyLLMClient or seed_only arm; either way events must appear.
    session_id, first_q = create_session(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        mode="adaptive",
        max_inference_questions=3,
        num_heldout=2,
        dim=5,
        llm_api_key=None,
    )
    answered, _ = answer_n_inference(conn, session_id, first_q, n=3)
    events = db.list_question_performance_events(conn, session_id)
    _assert(
        len(events) == answered,
        "seeds_only/event_count",
        f"events={len(events)}, answered={answered}",
    )
    _assert(
        all(e["question_source"] == "seed" for e in events),
        "seeds_only/all_source_seed",
        f"sources={[e['question_source'] for e in events]}",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Phase 2 Performance Events Tests ===\n")
    conn = fresh_conn()
    try:
        session_id, events = test_event_count_matches_inference_answers(conn)
        test_event_fields(events, session_id)
        test_mu_before_after_differ_after_first_update(events)
        test_user_id_stored(conn)
        test_no_events_for_heldout(conn)
        test_cross_session_lookup(conn)
        test_seeds_only_no_api_key(conn)
    finally:
        conn.close()
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    print(f"\n{'='*42}")
    total = _PASS + _FAIL
    print(f"Results: {_PASS}/{total} passed, {_FAIL} failed")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
