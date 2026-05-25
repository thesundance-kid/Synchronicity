#!/usr/bin/env python3
"""
Smoke tests for the three read-only query scripts.

Builds a temporary DB with a small mix of sessions, then calls the query
functions directly (no subprocess) and asserts that they return well-formed
data and don't crash.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db
from app import session_manager as sm
from scripts.query_question_stats import question_stats
from scripts.query_generated_candidates import generated_candidates_stats
from scripts.query_user_trajectory import session_trajectory, user_trajectory


QUESTIONS_PATH = str(PROJECT_ROOT / "data" / "questions_v2.json")

PASSED = 0
FAILED = 0


def check(label: str, cond: bool) -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}")


# ---------------------------------------------------------------------------
# Shared DB fixture
# ---------------------------------------------------------------------------

class _OneLLMClient:
    """Returns exactly one valid generated question — always accepted."""
    def generate(self, prompt: str) -> str:
        return json.dumps([
            {
                "id": "gen_smoke_0",
                "text": "Do you find it easy to stay focused during complex tasks?",
                "trait": "conscientiousness",
            }
        ])


def _build_fixture_db(path: str) -> dict:
    """
    Populate a fresh DB with:
      - 1 user
      - 2 seed_only sessions (user-linked, both completed)
      - 1 seed_plus_generated session (anonymous, DummyLLMClient)
    Returns a dict of IDs for later assertions.
    """
    conn = db.connect(path)
    db.init_db(conn)
    db.seed_question_parameters(conn, QUESTIONS_PATH)

    user_id = "smoke_user_001"
    db.insert_user(conn, user_id=user_id)

    ids = {"user_id": user_id, "seed_sessions": [], "gen_session": None}

    # Two completed seed_only sessions for the user.
    original_arm = sm.assign_experiment_arm
    sm.assign_experiment_arm = lambda sid: "seed_only"
    try:
        for _ in range(2):
            session_id, first_q = sm.create_session(
                conn,
                questions_v2_path=QUESTIONS_PATH,
                mode="fixed_order",
                max_inference_questions=3,
                num_heldout=2,
                dim=5,
                user_id=user_id,
            )
            # Answer all inference questions.
            q = first_q
            while q is not None and q.pool == "inference":
                result = sm.record_answer(
                    conn,
                    questions_v2_path=QUESTIONS_PATH,
                    session_id=session_id,
                    question_id=q.id,
                    response=3,
                    dim=5,
                )
                nq = result.get("next_question")
                if nq and nq.get("pool") == "inference":
                    from app.session_manager import QuestionPayload
                    q = QuestionPayload(**nq)
                else:
                    break
            # Answer heldout questions to complete.
            heldout_q = sm.get_next_question(conn, questions_v2_path=QUESTIONS_PATH, session_id=session_id, dim=5)
            while heldout_q is not None:
                sm.record_answer(
                    conn,
                    questions_v2_path=QUESTIONS_PATH,
                    session_id=session_id,
                    question_id=heldout_q.id,
                    response=3,
                    dim=5,
                )
                heldout_q = sm.get_next_question(conn, questions_v2_path=QUESTIONS_PATH, session_id=session_id, dim=5)

            ids["seed_sessions"].append(session_id)
    finally:
        sm.assign_experiment_arm = original_arm

    # One seed_plus_generated session (anonymous).
    sm.assign_experiment_arm = lambda sid: "seed_plus_generated"
    try:
        session_id, first_q = sm.create_session(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            mode="fixed_order",
            max_inference_questions=2,
            num_heldout=1,
            dim=5,
            llm_client=_OneLLMClient(),
        )
        sm.record_answer(
            conn,
            questions_v2_path=QUESTIONS_PATH,
            session_id=session_id,
            question_id=first_q.id,
            response=4,
            dim=5,
        )
        ids["gen_session"] = session_id
    finally:
        sm.assign_experiment_arm = original_arm

    conn.close()
    return ids


# ---------------------------------------------------------------------------
# Test: query_question_stats
# ---------------------------------------------------------------------------

def test_question_stats(db_path: str, ids: dict) -> None:
    print("\n[query_question_stats]")
    conn = db.connect(db_path)
    conn.row_factory = __import__("sqlite3").Row

    results = question_stats(conn, min_n=1, sort="n_answered")
    check("returns a list", isinstance(results, list))
    check("at least one question with events", len(results) > 0)

    if results:
        first = results[0]
        check("has question_id", "question_id" in first)
        check("has n_answered > 0", first["n_answered"] > 0)
        check("has mean_predicted_eig key", "mean_predicted_eig" in first)
        check("has mean_realized_ig key", "mean_realized_ig" in first)
        check("has response_dist dict", isinstance(first["response_dist"], dict))
        check("has param_version_counts dict", isinstance(first["param_version_counts"], dict))
        check("param_version has '1'", "1" in first["param_version_counts"])

    results_rig = question_stats(conn, min_n=1, sort="mean_rig")
    check("sort=mean_rig returns list", isinstance(results_rig, list))

    results_eig = question_stats(conn, min_n=999, sort="n_answered")
    check("min_n filter works (returns empty for high threshold)", results_eig == [])

    conn.close()


# ---------------------------------------------------------------------------
# Test: query_generated_candidates
# ---------------------------------------------------------------------------

def test_generated_candidates(db_path: str, ids: dict) -> None:
    print("\n[query_generated_candidates]")
    conn = db.connect(db_path)
    conn.row_factory = __import__("sqlite3").Row

    stats = generated_candidates_stats(conn)
    check("returns dict", isinstance(stats, dict))
    check("has total key", "total" in stats)
    check("has n_accepted key", "n_accepted" in stats)
    check("has n_dedupe_failed key", "n_dedupe_failed" in stats)
    check("has n_validation_failed key", "n_validation_failed" in stats)
    check("has n_selected key", "n_selected" in stats)
    check("has top_failure_reasons list", isinstance(stats["top_failure_reasons"], list))
    check("has per_session list", isinstance(stats["per_session"], list))
    check("has accepted_texts list", isinstance(stats["accepted_texts"], list))
    check("total >= 0", stats["total"] >= 0)
    check("n_accepted <= total", stats["n_accepted"] <= stats["total"])

    # Filter by the generated session.
    gen_sid = ids["gen_session"]
    if gen_sid:
        stats_filtered = generated_candidates_stats(conn, session_id=gen_sid)
        check("session filter returns dict", isinstance(stats_filtered, dict))
        check("session filter has total >= 0", stats_filtered["total"] >= 0)
        # all rows belong to the right session
        for s in stats_filtered["per_session"]:
            check(f"filtered session_id matches", s["session_id"] == gen_sid)

    conn.close()


# ---------------------------------------------------------------------------
# Test: query_user_trajectory (session mode)
# ---------------------------------------------------------------------------

def test_session_trajectory(db_path: str, ids: dict) -> None:
    print("\n[query_user_trajectory --session-id]")
    conn = db.connect(db_path)
    conn.row_factory = __import__("sqlite3").Row

    session_id = ids["seed_sessions"][0]
    data = session_trajectory(conn, session_id)

    check("returns dict", isinstance(data, dict))
    check("has session key", "session" in data)
    check("has snapshots key", "snapshots" in data)
    check("session_id matches", data["session"]["session_id"] == session_id)
    check("at least step_idx=0 snapshot exists", len(data["snapshots"]) >= 1)

    snaps = data["snapshots"]
    if snaps:
        check("snapshots ordered by step_idx", snaps[0]["step_idx"] == 0)
        check("entropy in first snapshot", "entropy" in snaps[0])
        check("mu_json in first snapshot", "mu_json" in snaps[0])
        check("entropy is float", isinstance(snaps[0]["entropy"], float))

    # Completed session should have multiple snapshots (step 0 + inference answers).
    check("completed session has >1 snapshot (prior + inference answers)", len(snaps) > 1)

    conn.close()


# ---------------------------------------------------------------------------
# Test: query_user_trajectory (user mode)
# ---------------------------------------------------------------------------

def test_user_trajectory(db_path: str, ids: dict) -> None:
    print("\n[query_user_trajectory --user-id]")
    conn = db.connect(db_path)
    conn.row_factory = __import__("sqlite3").Row

    user_id = ids["user_id"]
    data = user_trajectory(conn, user_id)

    check("returns dict", isinstance(data, dict))
    check("user_id matches", data["user_id"] == user_id)
    check("has user_created_at", data["user_created_at"] is not None)
    check("has current_state", data["current_state"] is not None)
    check("current_state has entropy", "entropy" in (data["current_state"] or {}))
    check("has completed_sessions list", isinstance(data["completed_sessions"], list))
    check("2 completed sessions", len(data["completed_sessions"]) == 2)
    check("has all_sessions list", isinstance(data["all_sessions"], list))
    check("all_sessions has 2 entries", len(data["all_sessions"]) == 2)
    check("session_numbers are 1, 2", [r["session_number"] for r in data["completed_sessions"]] == [1, 2])

    cur = data["current_state"]
    if cur:
        check("current_state has latest_session_id", "latest_session_id" in cur)
        check("current_state has mu_json", "mu_json" in cur)
        check("latest_step_idx > 0", cur.get("latest_step_idx", 0) > 0)

    snaps = data["latest_snapshots"]
    check("latest_snapshots is a list", isinstance(snaps, list))
    check("latest_snapshots non-empty", len(snaps) > 0)

    conn.close()


# ---------------------------------------------------------------------------
# Test: print functions don't crash (stdout output suppressed via redirect)
# ---------------------------------------------------------------------------

def test_print_functions_no_crash(db_path: str, ids: dict) -> None:
    print("\n[print functions smoke]")
    import io
    from scripts.query_question_stats import print_question_stats, _load_question_texts
    from scripts.query_generated_candidates import print_generated_candidates_stats
    from scripts.query_user_trajectory import print_session_trajectory, print_user_trajectory

    conn = db.connect(db_path)
    conn.row_factory = __import__("sqlite3").Row
    texts = _load_question_texts(QUESTIONS_PATH)

    deferred: list = []
    old_stdout = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        results = question_stats(conn, min_n=1)
        print_question_stats(results, texts, top_n=5)
        deferred.append(("print_question_stats ran", True))

        stats = generated_candidates_stats(conn)
        print_generated_candidates_stats(stats)
        deferred.append(("print_generated_candidates_stats ran", True))

        data_sess = session_trajectory(conn, ids["seed_sessions"][0])
        print_session_trajectory(data_sess)
        deferred.append(("print_session_trajectory ran", True))

        data_user = user_trajectory(conn, ids["user_id"])
        print_user_trajectory(data_user)
        deferred.append(("print_user_trajectory ran", True))
    except Exception as exc:
        deferred.append((f"no exception in print functions: {exc}", False))
    finally:
        output = buf.getvalue()
        sys.stdout = old_stdout

    for label, cond in deferred:
        check(label, cond)
    check("stdout output was non-empty", len(output) > 0)
    conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Query scripts smoke tests")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        print("\nBuilding fixture DB…")
        ids = _build_fixture_db(db_path)
        print(f"  user_id={ids['user_id']}")
        print(f"  seed sessions: {ids['seed_sessions']}")
        print(f"  gen session:   {ids['gen_session']}")

        test_question_stats(db_path, ids)
        test_generated_candidates(db_path, ids)
        test_session_trajectory(db_path, ids)
        test_user_trajectory(db_path, ids)
        test_print_functions_no_crash(db_path, ids)
    finally:
        Path(db_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 60)
    if FAILED > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
