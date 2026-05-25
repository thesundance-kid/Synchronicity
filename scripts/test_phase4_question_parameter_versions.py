#!/usr/bin/env python3
"""
Phase 4 smoke test: question_parameter_versions table, seed_question_parameters(),
insert/get/list helpers, and integration with record_answer (parameter_version field).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db
from app import session_manager as sm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_conn(tmp_path: str | Path):
    conn = db.connect(str(tmp_path))
    db.init_db(conn)
    return conn


def _questions_path() -> str:
    return str(PROJECT_ROOT / "data" / "questions_v2.json")


def _question_count(path: str) -> int:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return len(raw.get("inference_pool", [])) + len(raw.get("heldout_pool", []))
    return len(raw)


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
# Test 1: table exists after init_db
# ---------------------------------------------------------------------------

def test_table_exists() -> None:
    print("\n[1] Table exists after init_db")
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = _fresh_conn(f.name)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        check("question_parameter_versions table exists", "question_parameter_versions" in tables)
        conn.close()


# ---------------------------------------------------------------------------
# Test 2: seed_question_parameters inserts v1 for all 26 seed questions
# ---------------------------------------------------------------------------

def test_seed_creates_v1() -> None:
    print("\n[2] seed_question_parameters creates v1 for all seed questions")
    qpath = _questions_path()
    expected_count = _question_count(qpath)
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = _fresh_conn(f.name)
        db.seed_question_parameters(conn, qpath)
        total = conn.execute(
            "SELECT COUNT(*) FROM question_parameter_versions;"
        ).fetchone()[0]
        active_count = conn.execute(
            "SELECT COUNT(*) FROM question_parameter_versions WHERE active=1;"
        ).fetchone()[0]
        check(f"inserted {expected_count} rows", total == expected_count)
        check("all rows active=1", active_count == expected_count)
        check("all version=1", conn.execute(
            "SELECT COUNT(*) FROM question_parameter_versions WHERE version != 1;"
        ).fetchone()[0] == 0)
        check("all source='seed'", conn.execute(
            "SELECT COUNT(*) FROM question_parameter_versions WHERE source != 'seed';"
        ).fetchone()[0] == 0)
        conn.close()


# ---------------------------------------------------------------------------
# Test 3: seed_question_parameters is idempotent
# ---------------------------------------------------------------------------

def test_seed_idempotent() -> None:
    print("\n[3] seed_question_parameters is idempotent")
    qpath = _questions_path()
    expected_count = _question_count(qpath)
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = _fresh_conn(f.name)
        db.seed_question_parameters(conn, qpath)
        db.seed_question_parameters(conn, qpath)
        db.seed_question_parameters(conn, qpath)
        total = conn.execute(
            "SELECT COUNT(*) FROM question_parameter_versions;"
        ).fetchone()[0]
        check("no duplicate rows after 3 calls", total == expected_count)
        conn.close()


# ---------------------------------------------------------------------------
# Test 4: get_active_question_parameter_version returns v1 after seeding
# ---------------------------------------------------------------------------

def test_get_active_after_seeding() -> None:
    print("\n[4] get_active_question_parameter_version returns v1 after seeding")
    qpath = _questions_path()
    with open(qpath) as f:
        raw = json.load(f)
    first_q = raw["inference_pool"][0]
    qid = first_q["id"]

    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = _fresh_conn(f.name)
        db.seed_question_parameters(conn, qpath)
        active = db.get_active_question_parameter_version(conn, qid)
        check("active version is not None", active is not None)
        check("active version == 1", active is not None and active["version"] == 1)
        check("active flag is True", active is not None and active["active"] is True)
        check("source is 'seed'", active is not None and active["source"] == "seed")
        w = active["w"] if active else []
        check("w is a non-empty list", isinstance(w, list) and len(w) > 0)
        conn.close()


# ---------------------------------------------------------------------------
# Test 5: insert_question_parameter_version creates v2 and deactivates v1
# ---------------------------------------------------------------------------

def test_insert_version_deactivates_prior() -> None:
    print("\n[5] insert_question_parameter_version creates v2, deactivates v1")
    qpath = _questions_path()
    with open(qpath) as f:
        raw = json.load(f)
    q = raw["inference_pool"][0]
    qid = q["id"]
    w = q["w"]
    thr = q.get("thresholds", [-1.5, -0.5, 0.5, 1.5])

    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = _fresh_conn(f.name)
        db.seed_question_parameters(conn, qpath)

        v2 = db.insert_question_parameter_version(
            conn,
            question_id=qid,
            w=w,
            noise_var=0.8,
            thresholds=thr,
            source="estimated",
            estimation_method="vi",
            n_responses_used=50,
            performance_summary={"rmse": 0.12},
            active=True,
        )
        check("returned version == 2", v2 == 2)

        active = db.get_active_question_parameter_version(conn, qid)
        check("active version == 2", active is not None and active["version"] == 2)
        check("source is 'estimated'", active is not None and active["source"] == "estimated")

        all_versions = db.list_question_parameter_versions(conn, qid)
        check("list returns 2 versions", len(all_versions) == 2)
        check("v1 is inactive", all_versions[0]["version"] == 1 and all_versions[0]["active"] is False)
        check("v2 is active", all_versions[1]["version"] == 2 and all_versions[1]["active"] is True)
        check("n_responses_used stored", all_versions[1]["n_responses_used"] == 50)
        check("performance_summary stored", all_versions[1]["performance_summary"] == {"rmse": 0.12})
        conn.close()


# ---------------------------------------------------------------------------
# Test 6: list_question_parameter_versions returns versions in order
# ---------------------------------------------------------------------------

def test_list_versions_order() -> None:
    print("\n[6] list_question_parameter_versions returns all versions ASC")
    qpath = _questions_path()
    with open(qpath) as f:
        raw = json.load(f)
    q = raw["inference_pool"][1]
    qid = q["id"]
    w = q["w"]
    thr = q.get("thresholds", [-1.5, -0.5, 0.5, 1.5])

    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = _fresh_conn(f.name)
        db.seed_question_parameters(conn, qpath)
        db.insert_question_parameter_version(conn, question_id=qid, w=w, noise_var=0.9, thresholds=thr, source="estimated", active=True)
        db.insert_question_parameter_version(conn, question_id=qid, w=w, noise_var=0.8, thresholds=thr, source="estimated", active=True)
        versions = db.list_question_parameter_versions(conn, qid)
        check("3 versions in list", len(versions) == 3)
        check("versions are [1, 2, 3]", [v["version"] for v in versions] == [1, 2, 3])
        active_count = sum(1 for v in versions if v["active"])
        check("only 1 active version", active_count == 1)
        check("v3 is active", versions[2]["active"] is True)
        conn.close()


# ---------------------------------------------------------------------------
# Test 7: get_active returns None for unknown question_id
# ---------------------------------------------------------------------------

def test_get_active_unknown() -> None:
    print("\n[7] get_active_question_parameter_version returns None for unknown id")
    qpath = _questions_path()
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = _fresh_conn(f.name)
        db.seed_question_parameters(conn, qpath)
        result = db.get_active_question_parameter_version(conn, "nonexistent_q_id")
        check("returns None for unknown id", result is None)
        conn.close()


# ---------------------------------------------------------------------------
# Test 8: record_answer stores parameter_version=1 in performance events after seeding
# ---------------------------------------------------------------------------

class _DummySingleClient:
    """LLM client that always raises so the arm falls back to seed-only."""
    def generate(self, prompt: str) -> str:
        raise RuntimeError("no LLM in test")


def _make_seed_only_session(conn, qpath: str):
    original_arm = sm.assign_experiment_arm
    sm.assign_experiment_arm = lambda sid: "seed_only"
    try:
        session_id, first_q = sm.create_session(
            conn,
            questions_v2_path=qpath,
            mode="fixed_order",
            max_inference_questions=2,
            num_heldout=1,
            dim=5,
        )
    finally:
        sm.assign_experiment_arm = original_arm
    return session_id, first_q


def test_record_answer_stores_parameter_version() -> None:
    print("\n[8] record_answer stores parameter_version=1 for seeded questions")
    qpath = _questions_path()
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = _fresh_conn(f.name)
        db.seed_question_parameters(conn, qpath)

        session_id, first_q = _make_seed_only_session(conn, qpath)
        sm.record_answer(
            conn,
            questions_v2_path=qpath,
            session_id=session_id,
            question_id=first_q.id,
            response=3,
            dim=5,
        )

        events = conn.execute(
            "SELECT parameter_version FROM question_performance_events WHERE session_id=?;",
            (session_id,),
        ).fetchall()
        check("at least 1 performance event recorded", len(events) >= 1)
        check("parameter_version == 1", events[0][0] == 1)
        conn.close()


# ---------------------------------------------------------------------------
# Test 9: parameter_version=None when no active version exists (unseeded DB)
# ---------------------------------------------------------------------------

def test_record_answer_no_version_when_unseeded() -> None:
    print("\n[9] record_answer stores parameter_version=None in unseeded DB")
    qpath = _questions_path()
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = _fresh_conn(f.name)
        # Intentionally do NOT call seed_question_parameters.

        session_id, first_q = _make_seed_only_session(conn, qpath)
        sm.record_answer(
            conn,
            questions_v2_path=qpath,
            session_id=session_id,
            question_id=first_q.id,
            response=3,
            dim=5,
        )

        events = conn.execute(
            "SELECT parameter_version FROM question_performance_events WHERE session_id=?;",
            (session_id,),
        ).fetchall()
        check("at least 1 event", len(events) >= 1)
        check("parameter_version is None when not seeded", events[0][0] is None)
        conn.close()


# ---------------------------------------------------------------------------
# Test 10: generated questions (gen_ prefix) without versions do not crash
# ---------------------------------------------------------------------------

class _FixedOneLLMClient:
    """Always returns exactly one valid generated question."""
    def generate(self, prompt: str) -> str:
        return json.dumps([
            {"id": "gen_0", "text": "Do you enjoy meeting new people and socializing often?", "trait": "extraversion"}
        ])


def test_generated_question_no_version_no_crash() -> None:
    print("\n[10] Generated questions (no parameter_version) don't crash record_answer")
    qpath = _questions_path()
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = _fresh_conn(f.name)
        db.seed_question_parameters(conn, qpath)

        original_arm = sm.assign_experiment_arm
        sm.assign_experiment_arm = lambda sid: "seed_plus_generated"
        try:
            session_id, first_q = sm.create_session(
                conn,
                questions_v2_path=qpath,
                mode="fixed_order",
                max_inference_questions=2,
                num_heldout=1,
                dim=5,
                llm_client=_FixedOneLLMClient(),
            )
        finally:
            sm.assign_experiment_arm = original_arm

        # Answer whatever question was selected — could be seed or generated.
        try:
            sm.record_answer(
                conn,
                questions_v2_path=qpath,
                session_id=session_id,
                question_id=first_q.id,
                response=3,
                dim=5,
            )
            crashed = False
        except Exception as e:
            crashed = True
            print(f"    ERROR: {e}")

        check("no crash when answering after generated-arm session", not crashed)
        events = conn.execute(
            "SELECT question_id, parameter_version FROM question_performance_events WHERE session_id=?;",
            (session_id,),
        ).fetchall()
        check("at least 1 performance event", len(events) >= 1)
        conn.close()


# ---------------------------------------------------------------------------
# Test 11: Phases 1, 2, 3 tests still pass with seeded DB
# ---------------------------------------------------------------------------

def test_prior_phases_unaffected() -> None:
    print("\n[11] Phase 1/2/3 features work with seeded DB")
    qpath = _questions_path()
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = _fresh_conn(f.name)
        db.seed_question_parameters(conn, qpath)

        # Phase 1: user registration and warm-start.
        user_id = "test_user_phase4_x"
        db.insert_user(conn, user_id=user_id)
        user = db.get_user(conn, user_id)
        check("user registered", user is not None)

        original_arm = sm.assign_experiment_arm
        sm.assign_experiment_arm = lambda sid: "seed_only"
        try:
            session_id, first_q = sm.create_session(
                conn,
                questions_v2_path=qpath,
                mode="fixed_order",
                max_inference_questions=2,
                num_heldout=1,
                dim=5,
                user_id=user_id,
            )
        finally:
            sm.assign_experiment_arm = original_arm

        # Phase 2: performance event written with parameter_version=1.
        sm.record_answer(
            conn,
            questions_v2_path=qpath,
            session_id=session_id,
            question_id=first_q.id,
            response=3,
            dim=5,
        )

        events = conn.execute(
            "SELECT * FROM question_performance_events WHERE session_id=?;",
            (session_id,),
        ).fetchall()
        check("Phase 2 performance event present", len(events) >= 1)

        # Phase 1: posterior snapshot at step 0 and step 1.
        snapshots = db.list_posterior_snapshots(conn, session_id)
        check("Phase 1 posterior snapshots recorded", len(snapshots) >= 2)

        # Phase 1: user current state updated.
        state = db.get_user_current_state(conn, user_id)
        check("Phase 1 user_current_state updated", state is not None)

        # Phase 3: generated_question_candidates table still exists (even if empty for seed_only).
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()}
        check("Phase 3 table still present", "generated_question_candidates" in tables)
        conn.close()


# ---------------------------------------------------------------------------
# Test 12: seed-only / no-API-key path still works
# ---------------------------------------------------------------------------

def test_seed_only_no_api_key() -> None:
    print("\n[12] Seed-only / no-API-key path works end-to-end with Phase 4")
    qpath = _questions_path()
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        conn = _fresh_conn(f.name)
        db.seed_question_parameters(conn, qpath)

        original_arm = sm.assign_experiment_arm
        sm.assign_experiment_arm = lambda sid: "seed_only"
        try:
            session_id, first_q = sm.create_session(
                conn,
                questions_v2_path=qpath,
                mode="adaptive",
                max_inference_questions=3,
                num_heldout=2,
                dim=5,
                llm_api_key=None,
            )
        finally:
            sm.assign_experiment_arm = original_arm

        result = sm.record_answer(
            conn,
            questions_v2_path=qpath,
            session_id=session_id,
            question_id=first_q.id,
            response=4,
            dim=5,
        )
        check("record_answer returns dict", isinstance(result, dict))
        check("status is inference or heldout", result.get("status") in ("inference", "heldout", "complete"))

        events = conn.execute(
            "SELECT parameter_version FROM question_performance_events WHERE session_id=?;",
            (session_id,),
        ).fetchall()
        check("parameter_version=1 in seed-only run", len(events) >= 1 and events[0][0] == 1)
        conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 4 smoke test: question_parameter_versions")
    print("=" * 60)

    test_table_exists()
    test_seed_creates_v1()
    test_seed_idempotent()
    test_get_active_after_seeding()
    test_insert_version_deactivates_prior()
    test_list_versions_order()
    test_get_active_unknown()
    test_record_answer_stores_parameter_version()
    test_record_answer_no_version_when_unseeded()
    test_generated_question_no_version_no_crash()
    test_prior_phases_unaffected()
    test_seed_only_no_api_key()

    print("\n" + "=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 60)
    if FAILED > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
