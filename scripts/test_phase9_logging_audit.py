#!/usr/bin/env python3
"""
Phase 9 smoke tests: logging audit fixes.

Tests four priority fixes:
1. Missing indexes on QPE and GQC tables
2. Top-K candidate alternatives logged at each selection step
3. Raw LLM response text stored in llm_generation_requests
4. QPE.calibration_status stores actual status, not default 'candidate'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db
import app.session_manager as sm
from app.session_manager import SELECTION_LOG_TOP_K
from models.question_generation import DummyLLMClient

TEST_DB = str(PROJECT_ROOT / "data" / "pilot_phase9_test.db")
QUESTIONS = str(PROJECT_ROOT / "data" / "questions_v2.json")


def check(label: str, cond: bool, msg: str = "") -> None:
    if not cond:
        raise AssertionError(f"FAIL: {label}. {msg}")
    print(f"ok - {label}")


def fresh_conn():
    for suffix in ("", "-wal", "-shm"):
        p = TEST_DB + suffix
        if os.path.exists(p):
            os.remove(p)
    conn = db.connect(TEST_DB)
    db.init_db(conn)
    db.seed_question_parameters(conn, QUESTIONS)
    from models.prompt_policy import GENERIC_TEMPLATE
    db.seed_prompt_policies(conn, GENERIC_TEMPLATE)
    db.seed_exploratory_prompt_policies(conn)
    db.backfill_routing_enabled(conn)
    return conn


# ---------------------------------------------------------------------------
# 1. Schema: indexes
# ---------------------------------------------------------------------------

def test_indexes(conn) -> None:
    idx_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index';"
    ).fetchall()
    idx_names = {r[0] for r in idx_rows}
    check("index/idx_qpe_question_id", "idx_qpe_question_id" in idx_names)
    check("index/idx_qpe_question_created", "idx_qpe_question_created" in idx_names)
    check("index/idx_gqc_question_id", "idx_gqc_question_id" in idx_names)


# ---------------------------------------------------------------------------
# 2. Schema: new columns
# ---------------------------------------------------------------------------

def test_schema_new_columns(conn) -> None:
    ssl_cols = {r[1] for r in conn.execute("PRAGMA table_info(selection_score_logs);")}
    check("schema/selection_score_logs.candidate_rank", "candidate_rank" in ssl_cols)

    lgr_cols = {r[1] for r in conn.execute("PRAGMA table_info(llm_generation_requests);")}
    check("schema/llm_generation_requests.raw_response_text", "raw_response_text" in lgr_cols)


# ---------------------------------------------------------------------------
# 3. Top-K alternatives logging
# ---------------------------------------------------------------------------

def _run_session_and_get_first_step_logs(conn, strategy="anchored_exploratory"):
    sid, _ = sm.create_session(
        conn,
        questions_v2_path=QUESTIONS,
        mode="adaptive",
        max_inference_questions=5,
        num_heldout=2,
        llm_client=DummyLLMClient(),
        session_strategy=strategy,
    )
    logs = db.list_selection_score_logs(conn, sid)
    step0 = [l for l in logs if l["step_idx"] == 0]
    return sid, step0


def test_topk_logs_created_exploratory(conn) -> None:
    sid, step0 = _run_session_and_get_first_step_logs(conn, "anchored_exploratory")
    check("topk/at_least_one_log", len(step0) >= 1, f"step0={step0}")
    winners = [l for l in step0 if l["selected"]]
    alts = [l for l in step0 if not l["selected"]]
    check("topk/exactly_one_winner", len(winners) == 1, f"winners={winners}")
    check("topk/at_least_one_alternative", len(alts) >= 1, f"alts={alts}")
    check("topk/at_most_k_minus_1_alternatives", len(alts) <= SELECTION_LOG_TOP_K - 1,
          f"len(alts)={len(alts)}, K={SELECTION_LOG_TOP_K}")


def test_topk_winner_rank_is_zero(conn) -> None:
    sid, step0 = _run_session_and_get_first_step_logs(conn, "anchored_exploratory")
    winners = [l for l in step0 if l["selected"]]
    check("topk/winner_rank_0", all(w["candidate_rank"] == 0 for w in winners),
          str(winners))


def test_topk_alternative_ranks_sequential(conn) -> None:
    sid, step0 = _run_session_and_get_first_step_logs(conn, "anchored_exploratory")
    alts = sorted([l for l in step0 if not l["selected"]], key=lambda x: x["candidate_rank"])
    for i, alt in enumerate(alts):
        check(f"topk/alt_rank_{i+1}", alt["candidate_rank"] == i + 1,
              f"alt={alt}")


def test_topk_logs_created_classic_eig(conn) -> None:
    sid, step0 = _run_session_and_get_first_step_logs(conn, "classic_eig")
    winners = [l for l in step0 if l["selected"]]
    alts = [l for l in step0 if not l["selected"]]
    check("topk_eig/exactly_one_winner", len(winners) == 1)
    check("topk_eig/has_alternatives", len(alts) >= 1, f"alts={alts}")
    check("topk_eig/at_most_k_minus_1", len(alts) <= SELECTION_LOG_TOP_K - 1)


def test_topk_eig_winner_has_eig_field(conn) -> None:
    sid, step0 = _run_session_and_get_first_step_logs(conn, "classic_eig")
    winner = next(l for l in step0 if l["selected"])
    check("topk_eig/winner_eig_not_none",
          winner["expected_information_gain"] is not None)
    alts = [l for l in step0 if not l["selected"]]
    check("topk_eig/alt_eig_not_none",
          all(a["expected_information_gain"] is not None for a in alts),
          str(alts))


def test_topk_no_duplicate_on_repeated_get_next(conn) -> None:
    """Calling get_next_question twice for the same pending step must not duplicate logs."""
    sid, _ = sm.create_session(
        conn,
        questions_v2_path=QUESTIONS,
        mode="adaptive",
        max_inference_questions=5,
        num_heldout=2,
        llm_client=DummyLLMClient(),
        session_strategy="anchored_exploratory",
    )
    # Call get_next_question again — same step, same pending question
    sm.get_next_question(conn, questions_v2_path=QUESTIONS, session_id=sid, dim=5)
    sm.get_next_question(conn, questions_v2_path=QUESTIONS, session_id=sid, dim=5)

    logs = db.list_selection_score_logs(conn, sid)
    step0 = [l for l in logs if l["step_idx"] == 0]
    winners = [l for l in step0 if l["selected"]]
    check("topk/no_dup_winner", len(winners) == 1,
          f"Expected 1 winner row, got {len(winners)}")


def test_topk_multiple_steps(conn) -> None:
    """Top-K rows are created for every inference step."""
    sid, first_q = sm.create_session(
        conn,
        questions_v2_path=QUESTIONS,
        mode="adaptive",
        max_inference_questions=3,
        num_heldout=2,
        llm_client=DummyLLMClient(),
        session_strategy="anchored_exploratory",
    )
    # Answer step 0
    sm.record_answer(conn, questions_v2_path=QUESTIONS, session_id=sid,
                     question_id=first_q.id, response=3)
    sess = db.get_session(conn, sid)
    if sess.status == "inference":
        sm.record_answer(conn, questions_v2_path=QUESTIONS, session_id=sid,
                         question_id=sess.pending_question_id, response=3)

    logs = db.list_selection_score_logs(conn, sid)
    steps_with_winner = {l["step_idx"] for l in logs if l["selected"]}
    check("topk/two_steps_logged", len(steps_with_winner) >= 2,
          f"steps_with_winner={steps_with_winner}")


# ---------------------------------------------------------------------------
# 4. Raw LLM response text
# ---------------------------------------------------------------------------

def test_raw_response_stored_for_dummy_client(conn) -> None:
    sid, _ = sm.create_session(
        conn,
        questions_v2_path=QUESTIONS,
        mode="adaptive",
        max_inference_questions=5,
        num_heldout=2,
        llm_client=DummyLLMClient(),
        session_strategy="anchored_exploratory",
    )
    requests = db.list_llm_generation_requests_for_session(conn, sid)
    check("raw_text/at_least_one_request", len(requests) >= 1)
    req = requests[0]
    check("raw_text/raw_response_text_not_none", req["raw_response_text"] is not None,
          f"raw_response_text={req['raw_response_text']}")
    check("raw_text/raw_response_text_nonempty", len(req["raw_response_text"]) > 0)


def test_raw_response_contains_question_text(conn) -> None:
    """DummyLLMClient returns a numbered list; raw text should contain that content."""
    sid, _ = sm.create_session(
        conn,
        questions_v2_path=QUESTIONS,
        mode="adaptive",
        max_inference_questions=5,
        num_heldout=2,
        llm_client=DummyLLMClient(),
        session_strategy="anchored_exploratory",
    )
    requests = db.list_llm_generation_requests_for_session(conn, sid)
    raw = requests[0]["raw_response_text"]
    # DummyLLMClient returns lines starting with "1. ", "2. " etc.
    check("raw_text/contains_numbered_items", "1." in raw, f"raw={raw[:200]}")


class _FailingLLMClient:
    def complete(self, prompt: str) -> str:
        raise RuntimeError("Simulated API failure")


def test_raw_response_null_on_failure(conn) -> None:
    """Failed generation stores null raw_response_text and error_message."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sid, _ = sm.create_session(
            conn,
            questions_v2_path=QUESTIONS,
            mode="adaptive",
            max_inference_questions=5,
            num_heldout=2,
            llm_client=_FailingLLMClient(),
            session_strategy="anchored_exploratory",
        )
    requests = db.list_llm_generation_requests_for_session(conn, sid)
    check("raw_text/failure_has_request", len(requests) >= 1)
    req = requests[0]
    check("raw_text/failed_status", req["status"] == "failed", f"status={req['status']}")
    check("raw_text/failed_raw_text_null", req["raw_response_text"] is None,
          f"raw_response_text={req['raw_response_text']}")
    check("raw_text/failed_error_message_set",
          req["error_message"] is not None and len(req["error_message"]) > 0,
          f"error_message={req['error_message']}")


# ---------------------------------------------------------------------------
# 5. QPE.calibration_status
# ---------------------------------------------------------------------------

def test_qpe_calibration_status_seed(conn) -> None:
    """Seed questions should store 'calibrated' in QPE.calibration_status."""
    sid, first_q = sm.create_session(
        conn,
        questions_v2_path=QUESTIONS,
        mode="adaptive",
        max_inference_questions=5,
        num_heldout=2,
        llm_client=DummyLLMClient(),
        session_strategy="classic_eig",  # seed-only: all questions are seed
    )
    sess = db.get_session(conn, sid)
    sm.record_answer(conn, questions_v2_path=QUESTIONS, session_id=sid,
                     question_id=first_q.id, response=3)
    events = db.list_question_performance_events(conn, sid)
    check("qpe_cal/at_least_one_event", len(events) >= 1)
    ev = events[0]
    check("qpe_cal/seed_status_not_default_candidate",
          ev.get("calibration_status") != "candidate",
          f"calibration_status={ev.get('calibration_status')}")
    check("qpe_cal/seed_status_is_calibrated",
          ev.get("calibration_status") == "calibrated",
          f"calibration_status={ev.get('calibration_status')}")


def test_qpe_calibration_status_generated(conn) -> None:
    """Generated questions should store 'accepted_uncalibrated' in QPE.calibration_status."""
    sid, _ = sm.create_session(
        conn,
        questions_v2_path=QUESTIONS,
        mode="adaptive",
        max_inference_questions=8,
        num_heldout=2,
        llm_client=DummyLLMClient(),
        session_strategy="anchored_exploratory",
        max_anchor_questions=2,
        max_generated_probes=4,
    )
    sess = db.get_session(conn, sid)
    # Answer enough questions to get to a generated one (if any)
    answered = 0
    for _ in range(6):
        sess = db.get_session(conn, sid)
        if sess.status != "inference" or sess.pending_question_id is None:
            break
        sm.record_answer(conn, questions_v2_path=QUESTIONS, session_id=sid,
                         question_id=sess.pending_question_id, response=3)
        answered += 1

    events = db.list_question_performance_events(conn, sid)
    generated_events = [e for e in events if e["question_source"] == "generated"]
    if generated_events:
        for ev in generated_events:
            check("qpe_cal/generated_status_not_candidate",
                  ev["calibration_status"] != "candidate",
                  f"calibration_status={ev['calibration_status']}")
            check("qpe_cal/generated_status_is_accepted_uncalibrated",
                  ev["calibration_status"] == "accepted_uncalibrated",
                  f"calibration_status={ev['calibration_status']}")
    else:
        # No generated questions reached — still check seed events are correct
        seed_events = [e for e in events if e["question_source"] == "seed"]
        if seed_events:
            check("qpe_cal/seed_events_have_calibrated",
                  all(e["calibration_status"] == "calibrated" for e in seed_events),
                  str([e["calibration_status"] for e in seed_events]))
        print("ok - qpe_cal/generated_status_not_tested (no generated questions reached)")


def test_qpe_calibration_status_fallback(conn) -> None:
    """If inference pool has no calibration_status set, QPE should use 'candidate' fallback."""
    # Direct DB insert test — simulate a missing calibration_status
    import secrets
    import numpy as np
    sid = "test_fallback_" + secrets.token_urlsafe(4)
    db.insert_session(
        conn,
        session_id=sid,
        mode="adaptive",
        status="inference",
        step=0,
        max_inference_questions=5,
        asked_ids=[],
        heldout_ids=[],
        fixed_order_ids=None,
        posterior_mu=[0.0] * 5,
        posterior_sigma=np.eye(5).tolist(),
    )
    db.insert_question_performance_event(
        conn,
        question_id="q_test",
        session_id=sid,
        user_id=None,
        step_idx=0,
        question_source="seed",
        parameter_version=None,
        predicted_eig=0.1,
        entropy_before=5.0,
        entropy_after=4.9,
        realized_information_gain=0.1,
        response_value=3,
        mu_before=[0.0] * 5,
        sigma_before=np.eye(5).tolist(),
        mu_after=[0.1] * 5,
        sigma_after=np.eye(5).tolist(),
        calibration_status=None,  # should fall back to 'candidate'
    )
    events = db.list_question_performance_events(conn, sid)
    check("qpe_cal/fallback_to_candidate",
          events[0]["calibration_status"] == "candidate",
          f"calibration_status={events[0].get('calibration_status')}")


# ---------------------------------------------------------------------------
# 6. Existing tests still pass (regression check)
# ---------------------------------------------------------------------------

def test_policy_scoring_still_works(conn) -> None:
    """Basic policy scoring still works after Phase 9 changes."""
    from scripts.recompute_policy_scores import recompute_policy_scores
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Run a quick session to generate data
        sid, first_q = sm.create_session(
            conn,
            questions_v2_path=QUESTIONS,
            mode="adaptive",
            max_inference_questions=3,
            num_heldout=2,
            llm_client=DummyLLMClient(),
            session_strategy="anchored_exploratory",
        )
        for _ in range(5):
            sess = db.get_session(conn, sid)
            if sess.status == "complete" or sess.pending_question_id is None:
                break
            sm.record_answer(conn, questions_v2_path=QUESTIONS, session_id=sid,
                             question_id=sess.pending_question_id, response=3)

        # Try running policy score recompute (may find nothing to process)
        result = recompute_policy_scores(db_path=TEST_DB, dry_run=False)
        check("regression/policy_scoring_no_exception", True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    conn = fresh_conn()
    try:
        print("\n=== Phase 9 Logging Audit Tests ===\n")

        print("--- 1. Indexes ---")
        test_indexes(conn)

        print("\n--- 2. Schema columns ---")
        test_schema_new_columns(conn)

        print("\n--- 3. Top-K selection logging (exploratory) ---")
        test_topk_logs_created_exploratory(conn)
        test_topk_winner_rank_is_zero(conn)
        test_topk_alternative_ranks_sequential(conn)
        test_topk_no_duplicate_on_repeated_get_next(conn)
        test_topk_multiple_steps(conn)

        print("\n--- 4. Top-K selection logging (classic EIG) ---")
        test_topk_logs_created_classic_eig(conn)
        test_topk_eig_winner_has_eig_field(conn)

        print("\n--- 5. Raw LLM response text ---")
        test_raw_response_stored_for_dummy_client(conn)
        test_raw_response_contains_question_text(conn)
        test_raw_response_null_on_failure(conn)

        print("\n--- 6. QPE.calibration_status ---")
        test_qpe_calibration_status_seed(conn)
        test_qpe_calibration_status_generated(conn)
        test_qpe_calibration_status_fallback(conn)

        print("\n--- 7. Regression: policy scoring ---")
        test_policy_scoring_still_works(conn)

        print("\n=== All Phase 9 tests passed ===")
    finally:
        conn.close()
        for suffix in ("", "-wal", "-shm"):
            p = TEST_DB + suffix
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    main()
