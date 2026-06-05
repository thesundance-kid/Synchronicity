#!/usr/bin/env python3
"""
Offline question-parameter calibration.

Fits a regularized ordinal-probit loading vector for questions with enough
performance-event history. This is intentionally batch/offline: it never affects
an in-progress session, and it promotes a new active parameter version only when
quality gates pass.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import optimize, stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db

DEFAULT_DB = str(PROJECT_ROOT / "data" / "pilot.db")
METHOD = "regularized_ordinal_probit_v1"


def _neg_loglik(
    w: np.ndarray,
    *,
    theta: np.ndarray,
    y: np.ndarray,
    thresholds: np.ndarray,
    noise_var: float,
    prior_w: np.ndarray,
    l2: float,
) -> float:
    std = math.sqrt(float(noise_var))
    scores = theta @ w
    K = thresholds.size + 1
    total = 0.0
    for i, yi in enumerate(y.astype(int)):
        if yi == 1:
            lo = -np.inf
            hi = thresholds[0]
        elif yi == K:
            lo = thresholds[-1]
            hi = np.inf
        else:
            lo = thresholds[yi - 2]
            hi = thresholds[yi - 1]
        p = float(stats.norm.cdf((hi - scores[i]) / std) - stats.norm.cdf((lo - scores[i]) / std))
        total -= math.log(max(p, 1e-12))
    delta = w - prior_w
    total += 0.5 * float(l2) * float(delta @ delta)
    return float(total)


def _load_events(conn: sqlite3.Connection, question_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT response_value, mu_before_json
        FROM question_performance_events
        WHERE question_id = ?
        ORDER BY created_at ASC, id ASC;
        """,
        (question_id,),
    ).fetchall()
    return [
        {"response": int(r["response_value"]), "mu": json.loads(r["mu_before_json"])}
        for r in rows
    ]


def calibrate_question(
    conn: sqlite3.Connection,
    *,
    question_id: str,
    min_responses: int = 50,
    activate: bool = True,
    l2: float = 2.0,
) -> Dict[str, Any]:
    active = db.get_active_question_parameter_version(conn, question_id)
    if active is None:
        db.insert_question_calibration_run(
            conn,
            question_id=question_id,
            status="failed_no_active_parameters",
            n_responses=0,
            old_version=None,
            new_version=None,
            active_promoted=False,
            method=METHOD,
            diagnostics={},
        )
        return {"question_id": question_id, "status": "failed_no_active_parameters"}

    events = _load_events(conn, question_id)
    n = len(events)
    if n < min_responses:
        diagnostics = {"minimum_responses": min_responses, "observed_responses": n}
        db.insert_question_calibration_run(
            conn,
            question_id=question_id,
            status="insufficient_data",
            n_responses=n,
            old_version=active["version"],
            new_version=None,
            active_promoted=False,
            method=METHOD,
            diagnostics=diagnostics,
        )
        return {"question_id": question_id, "status": "insufficient_data", **diagnostics}

    theta = np.asarray([e["mu"] for e in events], dtype=np.float64)
    y = np.asarray([e["response"] for e in events], dtype=np.int64)
    prior_w = np.asarray(active["w"], dtype=np.float64).reshape(-1)
    thresholds = np.asarray(active["thresholds"], dtype=np.float64).reshape(-1)
    noise_var = float(active["noise_var"])
    if theta.shape[1] != prior_w.size:
        raise ValueError(f"theta dimension {theta.shape[1]} does not match w dimension {prior_w.size}")

    old_loss = _neg_loglik(
        prior_w,
        theta=theta,
        y=y,
        thresholds=thresholds,
        noise_var=noise_var,
        prior_w=prior_w,
        l2=l2,
    )
    result = optimize.minimize(
        lambda x: _neg_loglik(
            np.asarray(x, dtype=np.float64),
            theta=theta,
            y=y,
            thresholds=thresholds,
            noise_var=noise_var,
            prior_w=prior_w,
            l2=l2,
        ),
        prior_w,
        method="BFGS",
    )
    if not result.success:
        diagnostics = {"optimizer_message": str(result.message), "old_loss": old_loss}
        db.insert_question_calibration_run(
            conn,
            question_id=question_id,
            status="failed_optimizer",
            n_responses=n,
            old_version=active["version"],
            new_version=None,
            active_promoted=False,
            method=METHOD,
            diagnostics=diagnostics,
        )
        return {"question_id": question_id, "status": "failed_optimizer", **diagnostics}

    new_w = np.asarray(result.x, dtype=np.float64).reshape(-1)
    new_loss = _neg_loglik(
        new_w,
        theta=theta,
        y=y,
        thresholds=thresholds,
        noise_var=noise_var,
        prior_w=prior_w,
        l2=l2,
    )
    improvement = float(old_loss - new_loss)
    w_norm = float(np.linalg.norm(new_w))
    passes = bool(improvement > 0.0 and w_norm <= 3.0)
    status = "promoted" if passes and activate else "estimated_not_promoted"
    new_version: Optional[int] = None
    if passes or not activate:
        summary = {
            "old_loss": float(old_loss),
            "new_loss": float(new_loss),
            "improvement": improvement,
            "w_norm": w_norm,
            "minimum_responses": min_responses,
        }
        new_version = db.insert_question_parameter_version(
            conn,
            question_id=question_id,
            w=new_w.tolist(),
            noise_var=noise_var,
            thresholds=thresholds.tolist(),
            source="estimated",
            estimation_method=METHOD,
            n_responses_used=n,
            performance_summary=summary,
            active=bool(passes and activate),
        )
    diagnostics = {
        "old_loss": float(old_loss),
        "new_loss": float(new_loss),
        "improvement": improvement,
        "w_norm": w_norm,
        "passes_gates": passes,
    }
    db.insert_question_calibration_run(
        conn,
        question_id=question_id,
        status=status if new_version is not None else "failed_gates",
        n_responses=n,
        old_version=active["version"],
        new_version=new_version,
        active_promoted=bool(passes and activate),
        method=METHOD,
        diagnostics=diagnostics,
    )
    return {
        "question_id": question_id,
        "status": status if new_version is not None else "failed_gates",
        "new_version": new_version,
        **diagnostics,
    }


def calibrate_all(
    db_path: str,
    *,
    min_responses: int = 50,
    activate: bool = True,
) -> List[Dict[str, Any]]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        qids = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT question_id FROM question_performance_events ORDER BY question_id ASC;"
            ).fetchall()
        ]
        return [
            calibrate_question(conn, question_id=qid, min_responses=min_responses, activate=activate)
            for qid in qids
        ]
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate question measurement parameters")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--question-id")
    parser.add_argument("--min-responses", type=int, default=50)
    parser.add_argument("--no-activate", action="store_true")
    args = parser.parse_args()

    if args.question_id:
        conn = db.connect(args.db)
        db.init_db(conn)
        try:
            result = calibrate_question(
                conn,
                question_id=args.question_id,
                min_responses=args.min_responses,
                activate=not args.no_activate,
            )
            print(json.dumps(result, indent=2))
        finally:
            conn.close()
    else:
        print(json.dumps(calibrate_all(args.db, min_responses=args.min_responses, activate=not args.no_activate), indent=2))


if __name__ == "__main__":
    main()
