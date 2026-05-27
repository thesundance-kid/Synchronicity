#!/usr/bin/env python3
"""
Phase 5 smoke tests: prompt_policy_versions, llm_generation_requests,
and Phase 3 lineage (generation_request_id, prompt_policy_version_id).

Covers:
- prompt_policy_versions table exists
- llm_generation_requests table exists
- generated_question_candidates has generation_request_id, prompt_policy_version_id
- seed_prompt_policies creates the initial generic policy
- seed_prompt_policies is idempotent
- get_active_prompt_policy_version returns the seeded policy
- only one globally active policy at a time
- insert_prompt_policy_version creates a new version and deactivates old
- render_prompt_policy produces equivalent output to build_generation_prompt for generic policy
- generation request is logged for seed_plus_generated sessions
- generation request is NOT logged for seed_only sessions
- generated candidates have generation_request_id and prompt_policy_version_id set
- full lineage traceable: policy → request → candidate
- generation failure still logs a request with n_returned=0
- anonymous sessions: user_id=None in generation request
- linked sessions: correct user_id in generation request
- existing Phase 1–4 behavior unaffected
- query_prompt_policy_stats runs without mutating the DB
"""

from __future__ import annotations

import os
import secrets
import sys
import warnings
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db
import app.session_manager as sm
from app.session_manager import QuestionPayload, create_session, get_next_question, record_answer
from models.prompt_policy import GENERIC_TEMPLATE, PromptPolicy, render_prompt_policy
from models.question_generation import build_generation_prompt

QUESTIONS_PATH = str(PROJECT_ROOT / "data" / "questions_v2.json")
TEST_DB_PATH = str(PROJECT_ROOT / "data" / "pilot_phase5_test.db")

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_conn() -> db.sqlite3.Connection:
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    conn = db.connect(TEST_DB_PATH)
    db.init_db(conn)
    return conn


def _seed_both(conn) -> None:
    """Seed question parameters and the generic prompt policy (mirrors startup)."""
    db.seed_question_parameters(conn, QUESTIONS_PATH)
    db.seed_prompt_policies(conn, GENERIC_TEMPLATE)


def _force_generated(conn, llm_client, *, mode="adaptive", max_inf=3, num_heldout=2,
                     fixed_order_ids=None, user_id=None):
    orig = sm.assign_experiment_arm
    sm.assign_experiment_arm = lambda sid: "seed_plus_generated"
    try:
        sid, first_q = create_session(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            mode=mode,
            max_inference_questions=max_inf,
            num_heldout=num_heldout,
            dim=5,
            llm_client=llm_client,
            fixed_order_ids=fixed_order_ids,
            user_id=user_id,
        )
    finally:
        sm.assign_experiment_arm = orig
    return sid, first_q


def _force_seed_only(conn, *, max_inf=3, num_heldout=2):
    orig = sm.assign_experiment_arm
    sm.assign_experiment_arm = lambda sid: "seed_only"
    try:
        sid, first_q = create_session(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            mode="adaptive",
            max_inference_questions=max_inf,
            num_heldout=num_heldout,
            dim=5,
        )
    finally:
        sm.assign_experiment_arm = orig
    return sid, first_q


def _answer_n(conn, session_id, first_q, n, response=3):
    q: Optional[QuestionPayload] = first_q
    answered = 0
    while q is not None and q.pool == "inference" and answered < n:
        out = record_answer(
            conn, questions_v2_path=QUESTIONS_PATH,
            session_id=session_id, question_id=q.id, response=response, dim=5,
        )
        answered += 1
        nq = out["next_question"]
        q = QuestionPayload(**nq) if nq is not None else None
    return answered


class _SingleValidLLM:
    def complete(self, prompt: str) -> str:
        return "1. Do you enjoy solving mathematical puzzles in your spare time?"


class _ErrorLLM:
    def complete(self, prompt: str) -> str:
        raise RuntimeError("Simulated LLM failure")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tables_exist(conn):
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()}
    check("tables/prompt_policy_versions", "prompt_policy_versions" in tables)
    check("tables/llm_generation_requests", "llm_generation_requests" in tables)

    info = {row[1] for row in conn.execute(
        "PRAGMA table_info(generated_question_candidates);"
    ).fetchall()}
    check("tables/gqc_has_generation_request_id", "generation_request_id" in info)
    check("tables/gqc_has_prompt_policy_version_id", "prompt_policy_version_id" in info)


def test_seed_and_active_policy(conn):
    _seed_both(conn)
    rows = db.list_prompt_policy_versions(conn)
    check("seed/at_least_one_policy", len(rows) >= 1, f"got {len(rows)}")

    active = db.get_active_prompt_policy_version(conn)
    check("seed/active_policy_not_none", active is not None)
    if active:
        check("seed/active_name_generic", active["name"] == "generic",
              f"name={active['name']}")
        check("seed/active_version_1", active["version"] == 1,
              f"version={active['version']}")
        check("seed/active_strategy_generic", active["strategy_type"] == "generic")
        check("seed/active_conditioning_none", active["conditioning_mode"] == "none")
        check("seed/active_flag_true", bool(active["active"]))


def test_seed_idempotent(conn):
    db.seed_prompt_policies(conn, GENERIC_TEMPLATE)
    db.seed_prompt_policies(conn, GENERIC_TEMPLATE)
    db.seed_prompt_policies(conn, GENERIC_TEMPLATE)
    rows = db.list_prompt_policy_versions(conn)
    check("seed_idempotent/still_one_row", len(rows) == 1, f"got {len(rows)}")


def test_only_one_active(conn):
    _seed_both(conn)
    # Insert a second policy version; old must deactivate.
    db.insert_prompt_policy_version(
        conn,
        name="generic",
        prompt_template=GENERIC_TEMPLATE + "\n# v2",
        strategy_type="generic",
        conditioning_mode="none",
        active=True,
    )
    active_rows = [r for r in db.list_prompt_policy_versions(conn) if r["active"]]
    check("one_active/exactly_one", len(active_rows) == 1,
          f"active count={len(active_rows)}")
    check("one_active/newest_is_active", active_rows[0]["version"] == 2,
          f"version={active_rows[0]['version']}")


def test_render_prompt_matches_build_generation_prompt(conn):
    _seed_both(conn)
    # Always use v1 (the seeded generic policy) regardless of which version is currently active.
    rows = db.list_prompt_policy_versions(conn)
    policy_row = next((r for r in rows if r["name"] == "generic" and r["version"] == 1), None)
    assert policy_row is not None, "generic v1 policy not found"

    from models.question_bank import load_question_pools_v2
    inference_pool, _ = load_question_pools_v2(QUESTIONS_PATH, expected_dim=5)
    seeds = [{"id": q.id, "text": q.text, "w": q.w} for q in inference_pool]
    n = 10

    rendered = render_prompt_policy(PromptPolicy.from_row(policy_row), seeds, n)
    legacy = build_generation_prompt(seeds, n)

    check("render/matches_legacy_output", rendered == legacy,
          f"\nrendered ({len(rendered)} chars):\n{rendered[:120]}...\n"
          f"legacy   ({len(legacy)} chars):\n{legacy[:120]}...")


def test_generation_request_logged(conn):
    _seed_both(conn)
    session_id, _ = _force_generated(conn, _SingleValidLLM())

    reqs = db.list_llm_generation_requests_for_session(conn, session_id)
    check("gen_request/one_row", len(reqs) == 1, f"got {len(reqs)}")
    if reqs:
        r = reqs[0]
        check("gen_request/session_id_matches", r["session_id"] == session_id)
        check("gen_request/n_requested_positive", r["n_requested"] > 0,
              f"n_requested={r['n_requested']}")
        check("gen_request/n_returned_positive", r["n_returned"] > 0,
              f"n_returned={r['n_returned']}")
        check("gen_request/policy_version_id_set",
              r["prompt_policy_version_id"] is not None,
              f"prompt_policy_version_id={r['prompt_policy_version_id']}")
        check("gen_request/entropy_before_set",
              r["entropy_before"] is not None and r["entropy_before"] > 0)
        check("gen_request/posterior_mu_stored", r["posterior_mu"] is not None)
        check("gen_request/uncertainty_summary_stored",
              r["uncertainty_summary"] is not None)
        check("gen_request/prompt_rendered_stored",
              r["prompt_rendered"] is not None and len(r["prompt_rendered"]) > 50)


def test_no_request_for_seed_only(conn):
    _seed_both(conn)
    session_id, _ = _force_seed_only(conn)
    reqs = db.list_llm_generation_requests_for_session(conn, session_id)
    check("seed_only/no_generation_request", len(reqs) == 0, f"got {len(reqs)}")


def test_candidates_have_lineage(conn):
    _seed_both(conn)
    session_id, _ = _force_generated(conn, _SingleValidLLM())

    reqs = db.list_llm_generation_requests_for_session(conn, session_id)
    assert len(reqs) >= 1
    req_id = reqs[0]["id"]
    policy_vid = reqs[0]["prompt_policy_version_id"]

    cands = db.list_generated_question_candidates(conn, session_id)
    check("lineage/at_least_one_candidate", len(cands) >= 1)
    for c in cands:
        check(
            f"lineage/cand_{c['candidate_index']}/generation_request_id",
            c["generation_request_id"] == req_id,
            f"got {c['generation_request_id']}, want {req_id}",
        )
        check(
            f"lineage/cand_{c['candidate_index']}/prompt_policy_version_id",
            c["prompt_policy_version_id"] == policy_vid,
            f"got {c['prompt_policy_version_id']}, want {policy_vid}",
        )


def test_generation_failure_logs_request(conn):
    _seed_both(conn)
    orig = sm.assign_experiment_arm
    sm.assign_experiment_arm = lambda sid: "seed_plus_generated"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session_id, _ = create_session(
                conn,
                questions_v2_path=QUESTIONS_PATH,
                mode="adaptive",
                max_inference_questions=2,
                num_heldout=2,
                dim=5,
                llm_client=_ErrorLLM(),
            )
    finally:
        sm.assign_experiment_arm = orig

    reqs = db.list_llm_generation_requests_for_session(conn, session_id)
    check("gen_failure/request_still_logged", len(reqs) == 1, f"got {len(reqs)}")
    if reqs:
        check("gen_failure/n_returned_zero", reqs[0]["n_returned"] == 0,
              f"n_returned={reqs[0]['n_returned']}")


def test_anonymous_session_user_id_none(conn):
    _seed_both(conn)
    session_id, _ = _force_generated(conn, _SingleValidLLM())

    reqs = db.list_llm_generation_requests_for_session(conn, session_id)
    check("anon/request_exists", len(reqs) >= 1)
    if reqs:
        check("anon/user_id_is_none", reqs[0]["user_id"] is None,
              f"user_id={reqs[0]['user_id']}")


def test_linked_session_user_id_stored(conn):
    _seed_both(conn)
    uid = secrets.token_urlsafe(16)
    db.insert_user(conn, user_id=uid)

    session_id, _ = _force_generated(conn, _SingleValidLLM(), user_id=uid)
    reqs = db.list_llm_generation_requests_for_session(conn, session_id)
    check("linked/request_exists", len(reqs) >= 1)
    if reqs:
        check("linked/user_id_correct", reqs[0]["user_id"] == uid,
              f"got {reqs[0]['user_id']}, want {uid}")


def test_phase1_4_unaffected(conn):
    _seed_both(conn)
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

    # Phase 1: snapshot exists.
    snaps = db.list_posterior_snapshots(conn, session_id)
    check("phase1/step0_snapshot", len(snaps) >= 1 and snaps[0]["step_idx"] == 0)

    _answer_n(conn, session_id, first_q, n=3)

    # Phase 1: user_current_state updated.
    state = db.get_user_current_state(conn, uid)
    check("phase1/current_state_updated", state is not None)

    # Phase 2: performance events.
    events = db.list_question_performance_events(conn, session_id)
    check("phase2/3_events", len(events) == 3, f"got {len(events)}")

    # Phase 4: parameter_version populated.
    check("phase4/parameter_version_set",
          all(e["parameter_version"] == 1 for e in events),
          str([e["parameter_version"] for e in events]))


def test_query_script_readonly(conn):
    _seed_both(conn)
    _force_generated(conn, _SingleValidLLM())

    from scripts.query_prompt_policy_stats import query_prompt_policy_stats
    stats = query_prompt_policy_stats(TEST_DB_PATH)
    check("query_script/returns_list", isinstance(stats, list))
    check("query_script/at_least_one_entry", len(stats) >= 1, f"got {len(stats)}")
    if stats:
        s = stats[0]
        check("query_script/has_policy_id", "policy_id" in s)
        check("query_script/has_n_requests", "n_requests" in s)
        check("query_script/has_acceptance_rate", "acceptance_rate" in s)
        # Across all policies at least one must have a recorded request (prior tests added v2).
        check("query_script/n_requests_positive",
              any(row["n_requests"] >= 1 for row in stats),
              f"all policies have 0 requests: {[row['n_requests'] for row in stats]}")
        check("query_script/n_accepted_nonneg", s["n_accepted"] >= 0)

    # Verify DB not mutated: row counts must not change between two calls.
    count_before = conn.execute(
        "SELECT COUNT(*) FROM prompt_policy_versions;"
    ).fetchone()[0]
    query_prompt_policy_stats(TEST_DB_PATH)
    count_after = conn.execute(
        "SELECT COUNT(*) FROM prompt_policy_versions;"
    ).fetchone()[0]
    check("query_script/did_not_mutate", count_before == count_after,
          f"before={count_before}, after={count_after}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Phase 5 Prompt Policy Tests ===\n")

    conn = fresh_conn()
    try:
        test_tables_exist(conn)
        test_seed_and_active_policy(conn)
        test_seed_idempotent(conn)
        test_only_one_active(conn)
        test_render_prompt_matches_build_generation_prompt(conn)
        test_generation_request_logged(conn)
        test_no_request_for_seed_only(conn)
        test_candidates_have_lineage(conn)
        test_generation_failure_logs_request(conn)
        test_anonymous_session_user_id_none(conn)
        test_linked_session_user_id_stored(conn)
        test_phase1_4_unaffected(conn)
        test_query_script_readonly(conn)
    finally:
        conn.close()
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    print(f"\n{'='*40}")
    total = PASSED + FAILED
    print(f"Results: {PASSED}/{total} passed, {FAILED} failed")
    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()
