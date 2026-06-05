#!/usr/bin/env python3
"""
Phase 7 smoke tests: anchored exploratory selection, generated metadata,
offline calibration, and prompt-policy reward scoring.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db
import app.session_manager as sm
from models.personality_state import PersonalityState
from scripts.calibrate_question_parameters import calibrate_question
from scripts.score_prompt_policies import score_prompt_policies

TEST_DB = str(PROJECT_ROOT / "data" / "pilot_phase7_test.db")
QUESTIONS = str(PROJECT_ROOT / "data" / "questions_v2.json")


def check(label: str, cond: bool, msg: str = "") -> None:
    if not cond:
        raise AssertionError(f"{label} failed. {msg}")
    print(f"ok - {label}")


def fresh_conn():
    for suffix in ("", "-wal", "-shm"):
        path = TEST_DB + suffix
        if os.path.exists(path):
            os.remove(path)
    conn = db.connect(TEST_DB)
    db.init_db(conn)
    db.seed_question_parameters(conn, QUESTIONS)
    from models.prompt_policy import GENERIC_TEMPLATE

    db.seed_prompt_policies(conn, GENERIC_TEMPLATE)
    db.seed_exploratory_prompt_policies(conn)
    return conn


def test_schema(conn) -> None:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")}
    check("schema/selection_score_logs", "selection_score_logs" in tables)
    check("schema/question_calibration_runs", "question_calibration_runs" in tables)
    check("schema/prompt_policy_scores", "prompt_policy_scores" in tables)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(generated_question_candidates);")}
    for col in ("intended_contrast_json", "provisional_w_source", "calibration_status", "embedding_ref"):
        check(f"schema/generated/{col}", col in cols)


def test_anchored_exploratory_session(conn) -> str:
    original_arm = sm.assign_experiment_arm
    sm.assign_experiment_arm = lambda sid: "seed_plus_generated"
    try:
        session_id, first_q = sm.create_session(
            conn,
            questions_v2_path=QUESTIONS,
            mode="adaptive",
            max_inference_questions=8,
            num_heldout=2,
            dim=5,
            session_strategy="anchored_exploratory",
            max_anchor_questions=2,
            max_generated_probes=6,
        )
    finally:
        sm.assign_experiment_arm = original_arm

    sess = db.get_session(conn, session_id)
    check("session/default_total_len", sess.max_inference_questions + len(sess.heldout_ids) == 10)
    check("session/generated_present", len(sess.generated_question_ids) > 0)
    cands = db.list_generated_question_candidates(conn, session_id)
    accepted = [c for c in cands if c["accepted_into_pool"]]
    check("generated/metadata_present", bool(accepted))
    check("generated/calibration_status", accepted[0]["calibration_status"] == "accepted_uncalibrated")
    check("generated/provisional_source", accepted[0]["provisional_w_source"] == "embedding_knn_seed_average")

    q = first_q
    answered = []
    for _ in range(3):
        answered.append(q.id)
        out = sm.record_answer(
            conn,
            questions_v2_path=QUESTIONS,
            session_id=session_id,
            question_id=q.id,
            response=3,
            dim=5,
        )
        if out["next_question"] is None:
            break
        q = sm.QuestionPayload(**out["next_question"])

    sources = [sm._question_source(qid, db.get_session(conn, session_id)) for qid in answered[:2]]
    check("composition/first_two_are_anchors", sources == ["seed", "seed"], str(sources))
    logs = db.list_selection_score_logs(conn, session_id)
    check("selection/logs_written", len(logs) >= 3, str(logs))
    check("selection/components_present", logs[0]["semantic_novelty"] is not None)
    return session_id


def _insert_synthetic_qpe(conn, *, question_id: str, session_id: str, theta: np.ndarray, response: int, step: int) -> None:
    db.insert_question_performance_event(
        conn,
        question_id=question_id,
        session_id=session_id,
        user_id=None,
        step_idx=step,
        question_source="seed",
        parameter_version=1,
        predicted_eig=0.1,
        entropy_before=1.0,
        entropy_after=0.9,
        realized_information_gain=0.1,
        response_value=response,
        mu_before=theta.tolist(),
        sigma_before=np.eye(theta.size).tolist(),
        mu_after=theta.tolist(),
        sigma_after=np.eye(theta.size).tolist(),
    )


def test_calibration(conn) -> None:
    qid = "synthetic_calibration_q"
    prior_w = [0.05, 0, 0, 0, 0]
    true_w = np.array([1.0, 0, 0, 0, 0], dtype=float)
    db.insert_question_parameter_version(
        conn,
        question_id=qid,
        w=prior_w,
        noise_var=1.0,
        thresholds=[-1.5, -0.5, 0.5, 1.5],
        source="seed",
        active=True,
    )
    db.insert_session(
        conn,
        session_id="synthetic_session",
        mode="adaptive",
        status="inference",
        step=0,
        max_inference_questions=1,
        asked_ids=[],
        heldout_ids=["n_01"],
        fixed_order_ids=None,
        posterior_mu=[0, 0, 0, 0, 0],
        posterior_sigma=np.eye(5).tolist(),
    )
    rng = np.random.default_rng(123)
    thresholds = np.array([-1.5, -0.5, 0.5, 1.5])
    for i in range(60):
        theta = rng.normal(size=5)
        z = float(true_w @ theta + rng.normal(scale=0.25))
        response = int(np.searchsorted(thresholds, z, side="left") + 1)
        _insert_synthetic_qpe(conn, question_id=qid, session_id="synthetic_session", theta=theta, response=response, step=i)

    under = calibrate_question(conn, question_id=qid, min_responses=100)
    check("calibration/underpowered_not_promoted", under["status"] == "insufficient_data")
    result = calibrate_question(conn, question_id=qid, min_responses=50)
    check("calibration/promoted", result["status"] == "promoted", str(result))
    active = db.get_active_question_parameter_version(conn, qid)
    old_err = float(np.linalg.norm(np.array(prior_w) - true_w))
    new_err = float(np.linalg.norm(np.array(active["w"]) - true_w))
    check("calibration/closer_to_true_w", new_err < old_err, f"old={old_err}, new={new_err}")


def test_policy_scoring_readonly(conn) -> None:
    before = conn.execute("SELECT COUNT(*) FROM prompt_policy_versions;").fetchone()[0]
    rows = score_prompt_policies(TEST_DB)
    after = conn.execute("SELECT COUNT(*) FROM prompt_policy_versions;").fetchone()[0]
    check("policy_scoring/returns_rows", len(rows) >= 1)
    check("policy_scoring/readonly", before == after)
    check("policy_scoring/reward_present", "reward_score" in rows[0])


def main() -> None:
    conn = fresh_conn()
    try:
        test_schema(conn)
        test_anchored_exploratory_session(conn)
        test_calibration(conn)
        test_policy_scoring_readonly(conn)
    finally:
        conn.close()
    print("Phase 7 exploratory learning smoke tests passed")


if __name__ == "__main__":
    main()
