#!/usr/bin/env python3
"""
Phase 6 smoke tests: backend integrity hardening.

Covers:
1. Pending-question validation
   - answering a non-pending inference question is rejected
   - answering the correct pending question succeeds
   - duplicate answer (already-answered) is rejected
   - answering after session completion is rejected
2. Atomic answer recording
   - if an event-logging step raises mid-transaction, no partial response remains
3. Direct generated-candidate lineage in question_performance_events
   - prompt_policy_version → generation_request → generated_candidate → performance_event
     is directly traceable via foreign keys in question_performance_events
4. Parameter version freezing
   - create session (freezes param versions)
   - activate a new parameter version for a question
   - answer original pending question
   - performance_event records the original (frozen) version, not the new active one
5. Explicit LLM generation failure logging
   - failing LLM client logs status='failed' and error_message in llm_generation_requests
   - session still starts successfully (seed-only fallback)
6. Anthropic client timeout/error wrapping
   - AnthropicLLMClient raises RuntimeError on API failure (not raw SDK exception)
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db
import app.session_manager as sm
from app.session_manager import create_session, record_answer, get_next_question
from models.prompt_policy import GENERIC_TEMPLATE

QUESTIONS_PATH = str(PROJECT_ROOT / "data" / "questions_v2.json")
TEST_DB_PATH = str(PROJECT_ROOT / "data" / "pilot_phase6_test.db")

PASSED = 0
FAILED = 0


def check(label: str, cond: bool, msg: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}" + (f": {msg}" if msg else ""))


def fresh_conn():
    # Remove DB and any WAL/SHM journal files from previous runs.
    for suffix in ("", "-wal", "-shm"):
        p = TEST_DB_PATH + suffix
        if os.path.exists(p):
            os.remove(p)
    conn = db.connect(TEST_DB_PATH)
    db.init_db(conn)
    db.seed_question_parameters(conn, QUESTIONS_PATH)
    db.seed_prompt_policies(conn, GENERIC_TEMPLATE)
    return conn


def _start_session(conn, *, mode="adaptive", max_inf=3, num_heldout=2):
    """Helper: create a seed-only adaptive session."""
    from models.question_generation import DummyLLMClient
    sid, first_q = create_session(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        mode=mode,
        max_inference_questions=max_inf,
        num_heldout=num_heldout,
        llm_client=DummyLLMClient(),  # never generate — keep tests deterministic
    )
    return sid, first_q


# ---------------------------------------------------------------------------
# Section 1: Pending-question validation
# ---------------------------------------------------------------------------

def test_pending_question_enforcement():
    print("\n[1] Pending-question validation")
    conn = fresh_conn()

    sid, first_q = _start_session(conn)
    pending_id = first_q.id

    # Find a different inference question that is NOT the pending one.
    sess = db.get_session(conn, sid)
    pool_ids = [d["id"] for d in sess.inference_pool]
    other_ids = [qid for qid in pool_ids if qid != pending_id]
    check("inference pool has at least 2 items", len(other_ids) >= 1)

    if other_ids:
        wrong_id = other_ids[0]
        try:
            record_answer(conn, questions_v2_path=QUESTIONS_PATH, session_id=sid,
                          question_id=wrong_id, response=3)
            check("answering non-pending inference question raises ValueError", False,
                  "no exception raised")
        except ValueError as exc:
            check("answering non-pending inference question raises ValueError", True)
            check("error mentions pending question", "pending" in str(exc).lower(),
                  str(exc))

    # Answering the correct pending question should succeed.
    result = record_answer(conn, questions_v2_path=QUESTIONS_PATH, session_id=sid,
                           question_id=pending_id, response=3)
    check("answering the pending question succeeds", result is not None)
    check("result has session_id", result.get("session_id") == sid)

    # Trying to answer the same question again → duplicate row constraint.
    import sqlite3 as _sqlite3
    try:
        record_answer(conn, questions_v2_path=QUESTIONS_PATH, session_id=sid,
                      question_id=pending_id, response=3)
        check("duplicate answer rejected", False, "no exception raised")
    except (_sqlite3.IntegrityError, ValueError):
        check("duplicate answer rejected", True)

    conn.close()


def test_answer_after_completion():
    print("\n[2] Answer rejected after session completion")
    conn = fresh_conn()

    sid, _ = _start_session(conn, max_inf=1, num_heldout=1)

    # Answer inference question.
    sess = db.get_session(conn, sid)
    inf_q_id = sess.pending_question_id
    record_answer(conn, questions_v2_path=QUESTIONS_PATH, session_id=sid,
                  question_id=inf_q_id, response=3)

    # Answer heldout question to reach completion.
    sess2 = db.get_session(conn, sid)
    heldout_q_id = sess2.pending_question_id
    record_answer(conn, questions_v2_path=QUESTIONS_PATH, session_id=sid,
                  question_id=heldout_q_id, response=3)

    sess3 = db.get_session(conn, sid)
    check("session is complete after all answers", sess3.status == "complete")

    # Any further answer should raise ValueError.
    try:
        record_answer(conn, questions_v2_path=QUESTIONS_PATH, session_id=sid,
                      question_id=inf_q_id, response=3)
        check("answering after completion raises ValueError", False, "no exception")
    except ValueError as exc:
        check("answering after completion raises ValueError", True)
        check("error mentions complete", "complete" in str(exc).lower(), str(exc))

    conn.close()


# ---------------------------------------------------------------------------
# Section 2: Atomic answer recording
# ---------------------------------------------------------------------------

def test_atomic_answer_recording():
    print("\n[3] Atomic answer recording — no partial state on mid-transaction failure")
    conn = fresh_conn()

    sid, first_q = _start_session(conn)
    pending_id = first_q.id

    # Patch db.insert_question_performance_event to raise after insert_response has
    # been executed (but before the transaction is committed).
    original_qpe = db.insert_question_performance_event

    def _fail_qpe(*args, **kwargs):
        raise RuntimeError("simulated QPE failure")

    db.insert_question_performance_event = _fail_qpe
    try:
        try:
            record_answer(conn, questions_v2_path=QUESTIONS_PATH, session_id=sid,
                          question_id=pending_id, response=3)
            check("record_answer raises when QPE write fails", False, "no exception")
        except RuntimeError:
            check("record_answer raises when QPE write fails", True)
    finally:
        db.insert_question_performance_event = original_qpe

    # After rollback, no response row should exist.
    responses = db.list_responses(conn, sid)
    check("no response row committed after mid-transaction failure", len(responses) == 0,
          f"got {len(responses)} rows")

    # Session posterior should be unchanged (still step 0).
    sess = db.get_session(conn, sid)
    check("session step not incremented after rollback", sess.step == 0, f"step={sess.step}")

    # The pending question should still be set (cleared_pending happens inside transaction).
    check("pending_question_id still set after rollback",
          sess.pending_question_id == pending_id, sess.pending_question_id)

    # Now answer successfully to verify session is still usable.
    result = record_answer(conn, questions_v2_path=QUESTIONS_PATH, session_id=sid,
                           question_id=pending_id, response=3)
    check("session is usable after rollback recovery", result.get("status") in ("inference", "heldout"))

    conn.close()


# ---------------------------------------------------------------------------
# Section 3: Direct generated-candidate lineage in question_performance_events
# ---------------------------------------------------------------------------

def test_direct_lineage():
    print("\n[4] Direct QPE lineage: policy → request → candidate → performance_event")
    conn = fresh_conn()

    # Force a seed_plus_generated session by hooking into arm assignment.
    from models.session_experiment import assign_experiment_arm
    original_arm = sm.assign_experiment_arm

    def _force_generated(session_id):
        return "seed_plus_generated"

    sm.assign_experiment_arm = _force_generated
    try:
        from models.question_generation import DummyLLMClient
        sid, first_q = create_session(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            mode="adaptive",
            max_inference_questions=3,
            num_heldout=2,
            llm_client=DummyLLMClient(),
        )
    finally:
        sm.assign_experiment_arm = original_arm

    sess = db.get_session(conn, sid)
    gen_ids = list(sess.generated_question_ids)

    # Check Phase 6 columns exist on question_performance_events.
    cols = {row[1] for row in conn.execute(
        "PRAGMA table_info(question_performance_events);"
    ).fetchall()}
    check("question_performance_events has generated_candidate_id",
          "generated_candidate_id" in cols)
    check("question_performance_events has generation_request_id",
          "generation_request_id" in cols)
    check("question_performance_events has prompt_policy_version_id",
          "prompt_policy_version_id" in cols)

    if not gen_ids:
        print("  INFO  no generated questions accepted into pool (filtered by dedupe/validation); lineage checks skipped")
    else:
        # Force answering the first generated question directly (bypassing EIG) by
        # patching pending_question_id to point at a generated question.
        gen_qid = gen_ids[0]
        db.update_pending_selection(conn, session_id=sid, question_id=gen_qid, eig=0.1)

        record_answer(conn, questions_v2_path=QUESTIONS_PATH, session_id=sid,
                      question_id=gen_qid, response=3)

        # Verify direct lineage on the QPE row.
        events = db.list_question_performance_events(conn, sid)
        gen_events = [e for e in events if e["question_id"] == gen_qid]
        check("QPE row created for answered generated question", len(gen_events) > 0)
        if gen_events:
            evt = gen_events[-1]
            check("QPE generated_candidate_id is set for generated question",
                  evt.get("generated_candidate_id") is not None,
                  f"generated_candidate_id={evt.get('generated_candidate_id')}")
            check("QPE generation_request_id is set for generated question",
                  evt.get("generation_request_id") is not None,
                  f"generation_request_id={evt.get('generation_request_id')}")
            check("QPE prompt_policy_version_id is set for generated question",
                  evt.get("prompt_policy_version_id") is not None,
                  f"prompt_policy_version_id={evt.get('prompt_policy_version_id')}")

            # Verify the chain: QPE → candidate → request → policy.
            cand = db.get_generated_candidate_for_session_question(conn, sid, gen_qid)
            check("generated_candidate row found by id",
                  cand is not None and cand["id"] == evt["generated_candidate_id"])
            if cand:
                check("candidate generation_request_id matches QPE",
                      cand.get("generation_request_id") == evt["generation_request_id"])
                check("candidate prompt_policy_version_id matches QPE",
                      cand.get("prompt_policy_version_id") == evt["prompt_policy_version_id"])

    # QPE rows for seed questions should have NULL lineage fields.
    events = db.list_question_performance_events(conn, sid)
    seed_events = [e for e in events if e["question_source"] == "seed"]
    for e in seed_events:
        check("seed QPE has NULL generated_candidate_id",
              e.get("generated_candidate_id") is None)

    conn.close()


# ---------------------------------------------------------------------------
# Section 4: Parameter version freezing
# ---------------------------------------------------------------------------

def test_parameter_version_freezing():
    print("\n[5] Parameter version freezing")
    conn = fresh_conn()

    sid, first_q = _start_session(conn)
    pending_id = first_q.id

    # Get the version that was frozen at session creation.
    sess = db.get_session(conn, sid)
    frozen_version = None
    if sess.inference_pool:
        for d in sess.inference_pool:
            if d["id"] == pending_id:
                frozen_version = d.get("param_version")
                break
    check("param_version frozen in inference_pool at session creation",
          frozen_version is not None, f"got {frozen_version}")

    # Insert a new (v2) parameter version for the pending question, making it active.
    pv_row = db.get_active_question_parameter_version(conn, pending_id)
    if pv_row:
        new_version = db.insert_question_parameter_version(
            conn,
            question_id=pending_id,
            w=pv_row["w"],
            noise_var=pv_row["noise_var"],
            thresholds=pv_row["thresholds"],
            source="estimated",
            active=True,
        )
        check("new parameter version created and activated", new_version > frozen_version,
              f"new={new_version} frozen={frozen_version}")

        # Verify the currently active version differs from the frozen one.
        active_now = db.get_active_question_parameter_version(conn, pending_id)
        check("active version changed after insert",
              active_now["version"] == new_version, str(active_now))

        # Answer the pending question.
        record_answer(conn, questions_v2_path=QUESTIONS_PATH, session_id=sid,
                      question_id=pending_id, response=3)

        # Check that the QPE logged the frozen version (v1), not the new active one (v2).
        events = db.list_question_performance_events(conn, sid)
        matching = [e for e in events if e["question_id"] == pending_id]
        check("performance event exists for answered question", len(matching) > 0)
        if matching:
            logged_version = matching[0]["parameter_version"]
            check("QPE logs frozen parameter version (not new active)",
                  logged_version == frozen_version,
                  f"logged={logged_version} frozen={frozen_version} new_active={new_version}")
    else:
        check("active parameter version found for pending question (needed for test)", False,
              "no active parameter version found")

    conn.close()


# ---------------------------------------------------------------------------
# Section 5: Explicit LLM generation failure logging
# ---------------------------------------------------------------------------

def test_explicit_generation_failure():
    print("\n[6] Explicit LLM generation failure logging")
    conn = fresh_conn()

    # A client that always raises.
    class _FailingClient:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("test: simulated API error")

    from models.session_experiment import assign_experiment_arm
    original_arm = sm.assign_experiment_arm

    def _force_generated(session_id):
        return "seed_plus_generated"

    sm.assign_experiment_arm = _force_generated
    try:
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            sid, first_q = create_session(
                conn,
                questions_v2_path=QUESTIONS_PATH,
                mode="adaptive",
                max_inference_questions=3,
                num_heldout=2,
                llm_client=_FailingClient(),
            )
    finally:
        sm.assign_experiment_arm = original_arm

    check("session starts despite LLM failure", first_q is not None)

    # llm_generation_requests should have a row with status='failed'.
    gen_reqs = db.list_llm_generation_requests_for_session(conn, sid)
    check("generation request row created on failure", len(gen_reqs) >= 1,
          f"got {len(gen_reqs)} rows")
    if gen_reqs:
        req = gen_reqs[0]
        check("generation request status is 'failed'",
              req.get("status") == "failed", f"status={req.get('status')}")
        check("error_message is set",
              req.get("error_message") is not None and len(req["error_message"]) > 0,
              f"error_message={req.get('error_message')!r}")
        check("n_returned is 0 on failure", req.get("n_returned") == 0,
              f"n_returned={req.get('n_returned')}")

    # Ensure no generated candidates were logged (nothing to log on failure).
    cands = db.list_generated_question_candidates(conn, sid)
    check("no candidate rows on total failure", len(cands) == 0,
          f"got {len(cands)} candidate rows")

    conn.close()


def test_success_generation_status():
    print("\n[7] Successful generation logs status='success'")
    conn = fresh_conn()

    from models.session_experiment import assign_experiment_arm
    original_arm = sm.assign_experiment_arm

    def _force_generated(session_id):
        return "seed_plus_generated"

    sm.assign_experiment_arm = _force_generated
    try:
        from models.question_generation import DummyLLMClient
        sid, _ = create_session(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            mode="adaptive",
            max_inference_questions=3,
            num_heldout=2,
            llm_client=DummyLLMClient(),
        )
    finally:
        sm.assign_experiment_arm = original_arm

    gen_reqs = db.list_llm_generation_requests_for_session(conn, sid)
    check("generation request row exists on success", len(gen_reqs) >= 1)
    if gen_reqs:
        req = gen_reqs[0]
        check("generation request status is 'success'",
              req.get("status") == "success", f"status={req.get('status')}")
        check("error_message is None on success",
              req.get("error_message") is None, f"error_message={req.get('error_message')!r}")

    conn.close()


# ---------------------------------------------------------------------------
# Section 6: Anthropic client error wrapping
# ---------------------------------------------------------------------------

def test_anthropic_client_wraps_errors():
    print("\n[8] AnthropicLLMClient wraps errors as RuntimeError")
    from models.question_generation import AnthropicLLMClient
    import unittest.mock as mock

    # Patch the internal _client.messages.create to raise a generic exception.
    client = object.__new__(AnthropicLLMClient)
    client._model = "claude-haiku-4-5-20251001"
    mock_messages = mock.MagicMock()
    mock_messages.create.side_effect = Exception("connection refused")
    mock_anthropic = mock.MagicMock()
    mock_anthropic.messages = mock_messages
    client._client = mock_anthropic

    try:
        client.complete("test prompt")
        check("AnthropicLLMClient.complete raises RuntimeError on failure", False, "no exception")
    except RuntimeError as exc:
        check("AnthropicLLMClient.complete raises RuntimeError on failure", True)
        check("RuntimeError message contains original error type",
              "Exception" in str(exc) or "connection refused" in str(exc),
              str(exc))
    except Exception as exc:
        check("AnthropicLLMClient.complete raises RuntimeError on failure", False,
              f"got {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Section 7: Phase 6 columns exist after init_db
# ---------------------------------------------------------------------------

def test_schema_after_init():
    print("\n[9] Phase 6 schema columns present after init_db")
    conn = fresh_conn()

    qpe_cols = {row[1] for row in conn.execute(
        "PRAGMA table_info(question_performance_events);"
    ).fetchall()}
    check("QPE has generated_candidate_id", "generated_candidate_id" in qpe_cols)
    check("QPE has generation_request_id", "generation_request_id" in qpe_cols)
    check("QPE has prompt_policy_version_id (QPE)", "prompt_policy_version_id" in qpe_cols)

    lgr_cols = {row[1] for row in conn.execute(
        "PRAGMA table_info(llm_generation_requests);"
    ).fetchall()}
    check("LGR has status column", "status" in lgr_cols)
    check("LGR has error_message column", "error_message" in lgr_cols)

    conn.close()


# ---------------------------------------------------------------------------
# Section 8: Existing Phase 1–5 tests still pass (smoke regression)
# ---------------------------------------------------------------------------

def test_regression_core_flow():
    print("\n[10] Regression: core session flow still works")
    conn = fresh_conn()
    sid, first_q = _start_session(conn, max_inf=2, num_heldout=2)

    check("session created", sid is not None)
    check("first question has id", first_q.id is not None)
    check("first question is inference", first_q.pool == "inference")

    sess = db.get_session(conn, sid)
    check("session status is inference", sess.status == "inference")
    check("pending_question_id matches first_q", sess.pending_question_id == first_q.id)

    # Answer all inference questions.
    for _ in range(2):
        s = db.get_session(conn, sid)
        pid = s.pending_question_id
        if s.status == "inference" and pid:
            record_answer(conn, questions_v2_path=QUESTIONS_PATH, session_id=sid,
                          question_id=pid, response=3)

    sess2 = db.get_session(conn, sid)
    check("session transitions to heldout after all inference", sess2.status == "heldout")

    # Answer all heldout questions.
    for _ in range(2):
        s = db.get_session(conn, sid)
        pid = s.pending_question_id
        if s.status == "heldout" and pid:
            record_answer(conn, questions_v2_path=QUESTIONS_PATH, session_id=sid,
                          question_id=pid, response=3)

    sess3 = db.get_session(conn, sid)
    check("session completes after all answers", sess3.status == "complete")

    # Posterior snapshots: step 0 + 2 inference steps = 3 snapshots.
    snaps = db.list_posterior_snapshots(conn, sid)
    check("correct number of posterior snapshots", len(snaps) == 3, f"got {len(snaps)}")

    # Performance events: 2 inference answers.
    events = db.list_question_performance_events(conn, sid)
    check("2 performance events recorded", len(events) == 2, f"got {len(events)}")

    # All events have parameter_version set (seed questions should have v1).
    for e in events:
        check(f"performance event has parameter_version (step {e['step_idx']})",
              e["parameter_version"] is not None, str(e.get("parameter_version")))

    conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 6 Hardening Tests")
    print("=" * 60)

    test_schema_after_init()
    test_pending_question_enforcement()
    test_answer_after_completion()
    test_atomic_answer_recording()
    test_direct_lineage()
    test_parameter_version_freezing()
    test_explicit_generation_failure()
    test_success_generation_status()
    test_anthropic_client_wraps_errors()
    test_regression_core_flow()

    print()
    print("=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    # Cleanup test DB.
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass

    if FAILED > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
