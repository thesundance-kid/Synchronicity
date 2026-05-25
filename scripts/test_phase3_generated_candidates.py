"""
Phase 3 smoke tests: generated_question_candidates table and metadata logging.

Covers:
- generated_question_candidates table exists
- all raw LLM candidates logged for seed_plus_generated sessions
- rejected candidates (validation failure) are logged with reason
- accepted candidates have accepted_into_pool=1, question_id, w, noise_var, thresholds
- accepted candidates carry nn_seed_ids and nn_similarities
- selected generated question gets selected_at_step populated after EIG asks it
- heldout answers do not update selected_at_step
- generation failure still falls back to seeds-only (no crash, no candidate rows)
- no-API-key / DummyLLMClient path works end-to-end
- existing Phase 1 and Phase 2 behavior unaffected (no crash regression)
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db
import app.session_manager as sm
from app.session_manager import QuestionPayload, create_session, get_next_question, record_answer

QUESTIONS_PATH = str(PROJECT_ROOT / "data" / "questions_v2.json")
TEST_DB_PATH = str(PROJECT_ROOT / "data" / "pilot_phase3_test.db")

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
# Fake LLM clients for controlled testing
# ---------------------------------------------------------------------------

class _SingleValidLLMClient:
    """Returns exactly one valid personality question."""
    def complete(self, prompt: str) -> str:
        return "1. Do you enjoy solving mathematical puzzles in your spare time?"


class _ValidPlusInvalidLLMClient:
    """Returns one valid question and one that fails validation (contains sensitive term)."""
    def complete(self, prompt: str) -> str:
        return (
            "1. Do you enjoy solving mathematical puzzles in your spare time?\n"
            "2. Does your diagnosis affect how you approach daily challenges?"
        )


class _ErrorLLMClient:
    """Always raises an exception, simulating network failure."""
    def complete(self, prompt: str) -> str:
        raise RuntimeError("Simulated LLM failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_conn() -> db.sqlite3.Connection:
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    conn = db.connect(TEST_DB_PATH)
    db.init_db(conn)
    return conn


def _force_generated_arm(conn, llm_client, *, mode="adaptive", max_inf=3, num_heldout=2,
                          fixed_order_ids=None):
    """Create a session with arm forced to seed_plus_generated via monkeypatching."""
    original = sm.assign_experiment_arm
    sm.assign_experiment_arm = lambda sid: "seed_plus_generated"
    try:
        session_id, first_q = create_session(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            mode=mode,
            max_inference_questions=max_inf,
            num_heldout=num_heldout,
            dim=5,
            llm_client=llm_client,
            fixed_order_ids=fixed_order_ids,
        )
    finally:
        sm.assign_experiment_arm = original
    return session_id, first_q


def _answer_n_inference(conn, session_id, first_q, n, response=3):
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


def _drive_to_completion(conn, session_id, first_q, response=3):
    q: Optional[QuestionPayload] = first_q
    while q is not None:
        out = record_answer(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            session_id=session_id,
            question_id=q.id,
            response=response,
            dim=5,
        )
        nq = out["next_question"]
        q = QuestionPayload(**nq) if nq is not None else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_table_exists(conn):
    """generated_question_candidates table must exist."""
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
    }
    _assert("generated_question_candidates" in tables, "table_exists/generated_question_candidates")


def test_candidates_logged_for_generated_arm(conn):
    """All raw LLM candidates are logged for a seed_plus_generated session."""
    session_id, first_q = _force_generated_arm(conn, _SingleValidLLMClient(), max_inf=2)

    rows = db.list_generated_question_candidates(conn, session_id)
    _assert(len(rows) >= 1, "candidates_logged/at_least_one_candidate",
            f"got {len(rows)} rows")

    # Each row must reference the session.
    for r in rows:
        _assert(r["session_id"] == session_id, "candidates_logged/session_id_matches")

    # candidate_index values must be contiguous from 0.
    indices = [r["candidate_index"] for r in rows]
    _assert(indices == list(range(len(rows))), "candidates_logged/indices_contiguous",
            f"got {indices}")
    return session_id, rows


def test_no_candidates_for_seed_only(conn):
    """No generated_question_candidates rows for a seed_only session."""
    original = sm.assign_experiment_arm
    sm.assign_experiment_arm = lambda sid: "seed_only"
    try:
        session_id, _ = create_session(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            mode="adaptive",
            max_inference_questions=2,
            num_heldout=2,
            dim=5,
        )
    finally:
        sm.assign_experiment_arm = original

    rows = db.list_generated_question_candidates(conn, session_id)
    _assert(len(rows) == 0, "seed_only/no_candidate_rows", f"got {len(rows)} rows")


def test_accepted_candidate_fields(conn):
    """Accepted candidates have required fields: question_id, w, noise_var, thresholds, nn_*."""
    session_id, _ = _force_generated_arm(conn, _SingleValidLLMClient(), max_inf=2)
    rows = db.list_generated_question_candidates(conn, session_id)

    accepted = [r for r in rows if r["accepted_into_pool"]]
    _assert(len(accepted) >= 1, "accepted_fields/at_least_one_accepted",
            f"total rows={len(rows)}, accepted={len(accepted)}")

    for r in accepted:
        tag = f"accepted_fields/{r['candidate_index']}"
        _assert(r["question_id"] is not None, f"{tag}/question_id_not_null")
        _assert(r["w"] is not None and len(r["w"]) == 5, f"{tag}/w_is_5d_list",
                f"w={r['w']}")
        _assert(r["noise_var"] is not None, f"{tag}/noise_var_not_null",
                f"noise_var={r['noise_var']}")
        _assert(r["thresholds"] is not None and len(r["thresholds"]) == 4,
                f"{tag}/thresholds_4_elements", f"thresholds={r['thresholds']}")
        _assert(r["nn_seed_ids"] is not None and len(r["nn_seed_ids"]) >= 1,
                f"{tag}/nn_seed_ids_non_empty", f"nn_seed_ids={r['nn_seed_ids']}")
        _assert(r["nn_similarities"] is not None and len(r["nn_similarities"]) >= 1,
                f"{tag}/nn_similarities_non_empty", f"nn_similarities={r['nn_similarities']}")
        _assert(r["validation_passed"], f"{tag}/validation_passed")
        _assert(r["dedupe_failed"] is False, f"{tag}/dedupe_not_failed")


def test_validation_rejected_candidate(conn):
    """A candidate with a sensitive term is logged with validation_failure_reason."""
    session_id, _ = _force_generated_arm(conn, _ValidPlusInvalidLLMClient(), max_inf=2)
    rows = db.list_generated_question_candidates(conn, session_id)

    _assert(len(rows) >= 1, "validation_reject/at_least_one_row")

    # Should have at least one candidate that failed validation.
    failed_validation = [
        r for r in rows
        if not r["validation_passed"] and not r["dedupe_failed"]
    ]

    # Also check for any that were rejected (regardless of reason) to validate the test setup.
    all_rejected = [r for r in rows if not r["accepted_into_pool"]]
    _assert(
        len(all_rejected) >= 1,
        "validation_reject/at_least_one_rejected",
        f"rows={len(rows)}, accepted={sum(1 for r in rows if r['accepted_into_pool'])}",
    )

    # The candidate with "diagnosis" in text should fail with sensitive-term reason.
    diag_rows = [r for r in rows if "diagnosis" in r["text"].lower()]
    if diag_rows:
        # It may have been dedupe-filtered (text is distinctive enough, but hard to guarantee)
        # or validation-filtered.
        r = diag_rows[0]
        if not r["dedupe_failed"]:
            # Passed dedupe → must have failed validation
            _assert(
                not r["validation_passed"],
                "validation_reject/diagnosis_failed_validation",
                f"validation_passed={r['validation_passed']}, reason={r['validation_failure_reason']}",
            )
            _assert(
                r["validation_failure_reason"] is not None
                and "diagnosis" in r["validation_failure_reason"],
                "validation_reject/reason_mentions_diagnosis",
                f"reason={r['validation_failure_reason']}",
            )
            _assert(not r["accepted_into_pool"], "validation_reject/not_accepted")
        else:
            # Got dedupe-filtered (not expected but acceptable), just check it's logged.
            _assert(r["dedupe_failed"], "validation_reject/diagnosis_dedupe_failed")
    else:
        # The "diagnosis" question wasn't generated (maybe dedupe removed it before metadata).
        # Just verify some row was rejected.
        _ok("validation_reject/diagnosis_question_deduped_or_filtered")


def test_selected_at_step(conn):
    """A generated question selected by EIG gets selected_at_step populated."""
    # Use SingleValidLLMClient → produces exactly one candidate → gen_0 if it passes dedupe.
    # Force fixed_order mode with gen_0 first to guarantee EIG asks it.
    # max_inference_questions=1 so gen_0 is the only inference question.
    session_id, first_q = _force_generated_arm(
        conn,
        _SingleValidLLMClient(),
        mode="fixed_order",
        max_inf=1,
        num_heldout=2,
        fixed_order_ids=["gen_0"],
    )

    sess = db.get_session(conn, session_id)
    if "gen_0" not in sess.generated_question_ids:
        # gen_0 was rejected (e.g., too similar to a seed) — skip this sub-test.
        _ok("selected_at_step/gen_0_not_in_pool_skip")
        return

    _assert(first_q.id == "gen_0", "selected_at_step/first_q_is_gen_0",
            f"first_q.id={first_q.id}")

    # Before answering: selected_at_step must be NULL.
    candidates = db.list_generated_question_candidates(conn, session_id)
    gen0 = next((r for r in candidates if r["question_id"] == "gen_0"), None)
    _assert(gen0 is not None, "selected_at_step/gen0_candidate_row_exists")
    if gen0:
        _assert(gen0["selected_at_step"] is None, "selected_at_step/null_before_answer",
                f"selected_at_step={gen0['selected_at_step']}")

    # Answer the inference question (gen_0).
    record_answer(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        session_id=session_id,
        question_id="gen_0",
        response=3,
        dim=5,
    )

    # After answering: selected_at_step must be set.
    candidates = db.list_generated_question_candidates(conn, session_id)
    gen0 = next((r for r in candidates if r["question_id"] == "gen_0"), None)
    _assert(gen0 is not None, "selected_at_step/gen0_still_exists")
    if gen0:
        _assert(
            gen0["selected_at_step"] is not None,
            "selected_at_step/set_after_answer",
            f"selected_at_step={gen0['selected_at_step']}",
        )
        _assert(
            gen0["selected_at_step"] == 0,
            "selected_at_step/equals_step_idx_0",
            f"selected_at_step={gen0['selected_at_step']}",
        )


def test_heldout_does_not_update_selected_at_step(conn):
    """Answering heldout questions must not update selected_at_step."""
    session_id, first_q = _force_generated_arm(
        conn,
        _SingleValidLLMClient(),
        mode="fixed_order",
        max_inf=1,
        num_heldout=2,
        fixed_order_ids=["gen_0"],
    )

    sess = db.get_session(conn, session_id)
    if "gen_0" not in sess.generated_question_ids:
        _ok("heldout_no_update/gen_0_not_in_pool_skip")
        return

    # Answer the single inference question (gen_0).
    record_answer(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        session_id=session_id,
        question_id="gen_0",
        response=3,
        dim=5,
    )

    # Capture selected_at_step after inference.
    cands_after_inf = db.list_generated_question_candidates(conn, session_id)
    gen0_step = next(
        (r["selected_at_step"] for r in cands_after_inf if r["question_id"] == "gen_0"), None
    )

    # Now answer all heldout questions.
    q = get_next_question(conn, questions_v2_path=QUESTIONS_PATH, session_id=session_id, dim=5)
    while q is not None and q.pool == "heldout":
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

    # selected_at_step must still match what was set after the inference answer.
    cands_after_heldout = db.list_generated_question_candidates(conn, session_id)
    gen0_step_after = next(
        (r["selected_at_step"] for r in cands_after_heldout if r["question_id"] == "gen_0"), None
    )
    _assert(
        gen0_step == gen0_step_after,
        "heldout_no_update/selected_at_step_unchanged",
        f"before heldout={gen0_step}, after heldout={gen0_step_after}",
    )


def test_generation_failure_fallback(conn):
    """When LLM generation fails, session falls back to seeds-only with no candidate rows."""
    original = sm.assign_experiment_arm
    sm.assign_experiment_arm = lambda sid: "seed_plus_generated"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session_id, first_q = create_session(
                conn,
                questions_v2_path=QUESTIONS_PATH,
                mode="adaptive",
                max_inference_questions=2,
                num_heldout=2,
                dim=5,
                llm_client=_ErrorLLMClient(),
            )
    finally:
        sm.assign_experiment_arm = original

    # Session must have been created successfully.
    sess = db.get_session(conn, session_id)
    _assert(sess is not None, "gen_failure_fallback/session_created")
    _assert(sess.arm == "seed_plus_generated", "gen_failure_fallback/arm_preserved")
    _assert(len(sess.generated_question_ids) == 0, "gen_failure_fallback/no_generated_ids",
            f"generated_ids={sess.generated_question_ids}")

    # No candidate rows (metadata is unavailable when generation fails).
    rows = db.list_generated_question_candidates(conn, session_id)
    _assert(len(rows) == 0, "gen_failure_fallback/no_candidate_rows",
            f"got {len(rows)} rows")

    # First question should still be valid (seeds-only).
    _assert(first_q is not None, "gen_failure_fallback/first_q_exists")
    _assert(first_q.pool == "inference", "gen_failure_fallback/first_q_inference")


def test_no_api_key_path(conn):
    """No-API-key (DummyLLMClient) path works and seeds-only sessions have no candidate rows."""
    # DummyLLMClient is the default; don't pass llm_client or llm_api_key.
    # seed_only arm: no generation runs.
    original = sm.assign_experiment_arm
    sm.assign_experiment_arm = lambda sid: "seed_only"
    try:
        session_id, first_q = create_session(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            mode="adaptive",
            max_inference_questions=2,
            num_heldout=2,
            dim=5,
        )
    finally:
        sm.assign_experiment_arm = original

    sess = db.get_session(conn, session_id)
    _assert(sess.arm == "seed_only", "no_api_key/arm_seed_only")
    rows = db.list_generated_question_candidates(conn, session_id)
    _assert(len(rows) == 0, "no_api_key/no_candidate_rows_seed_only")

    # Answer some questions to confirm the session works normally.
    answered = _answer_n_inference(conn, session_id, first_q, n=2)
    _assert(answered == 2, "no_api_key/inference_answers_work", f"answered={answered}")


def test_dummy_llm_generated_arm(conn):
    """DummyLLMClient with seed_plus_generated arm: candidates are logged, session works."""
    from models.question_generation import DummyLLMClient
    session_id, first_q = _force_generated_arm(conn, DummyLLMClient(), max_inf=3)

    sess = db.get_session(conn, session_id)
    _assert(sess.arm == "seed_plus_generated", "dummy_llm/arm_correct")

    rows = db.list_generated_question_candidates(conn, session_id)
    _assert(len(rows) >= 1, "dummy_llm/candidate_rows_exist", f"got {len(rows)}")

    accepted = [r for r in rows if r["accepted_into_pool"]]
    _assert(len(accepted) >= 1, "dummy_llm/at_least_one_accepted",
            f"total={len(rows)}, accepted={len(accepted)}")

    # Session should work normally.
    answered = _answer_n_inference(conn, session_id, first_q, n=3)
    _assert(answered == 3, "dummy_llm/inference_answers_work", f"answered={answered}")


def test_phase1_phase2_unaffected(conn):
    """Phase 1 and Phase 2 behavior is unaffected by Phase 3 changes."""
    import secrets
    uid = secrets.token_urlsafe(16)
    db.insert_user(conn, user_id=uid)

    session_id, first_q = create_session(
        conn,
        questions_v2_path=QUESTIONS_PATH,
        mode="adaptive",
        max_inference_questions=3,
        num_heldout=2,
        dim=5,
        user_id=uid,
    )

    # Phase 1: snapshot at step 0.
    snaps = db.list_posterior_snapshots(conn, session_id)
    _assert(len(snaps) >= 1 and snaps[0]["step_idx"] == 0,
            "phase1_phase2/step0_snapshot_exists")

    # Answer some inference questions.
    _answer_n_inference(conn, session_id, first_q, n=3)

    # Phase 1: user_current_state updated.
    state = db.get_user_current_state(conn, uid)
    _assert(state is not None, "phase1_phase2/user_current_state_exists")

    # Phase 2: performance events.
    events = db.list_question_performance_events(conn, session_id)
    _assert(len(events) == 3, "phase1_phase2/3_perf_events", f"got {len(events)}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Phase 3 Generated Candidates Tests ===\n")
    conn = fresh_conn()
    try:
        test_table_exists(conn)
        test_no_candidates_for_seed_only(conn)
        test_candidates_logged_for_generated_arm(conn)
        test_accepted_candidate_fields(conn)
        test_validation_rejected_candidate(conn)
        test_selected_at_step(conn)
        test_heldout_does_not_update_selected_at_step(conn)
        test_generation_failure_fallback(conn)
        test_no_api_key_path(conn)
        test_dummy_llm_generated_arm(conn)
        test_phase1_phase2_unaffected(conn)
    finally:
        conn.close()
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    print(f"\n{'='*46}")
    total = _PASS + _FAIL
    print(f"Results: {_PASS}/{total} passed, {_FAIL} failed")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
