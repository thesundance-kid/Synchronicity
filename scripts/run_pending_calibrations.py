#!/usr/bin/env python3
"""
Run pending calibration jobs.

Finds calibration_jobs with status='pending', fits regularized ordinal-probit
loading vectors from accumulated question_performance_events, and promotes a new
question_parameter_version when all quality gates pass.

Run BEFORE recompute_policy_scores.py so that realized-IG in QPE reflects
better-calibrated w vectors before policy reward scores are refreshed.

Usage:
  python scripts/run_pending_calibrations.py
  python scripts/run_pending_calibrations.py --dry-run
  python scripts/run_pending_calibrations.py --question-id q_01 --min-responses 30
  python scripts/run_pending_calibrations.py --limit 5 --force
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import optimize, stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db

DEFAULT_DB = str(PROJECT_ROOT / "data" / "pilot.db")
METHOD = "regularized_ordinal_probit_v1"
_LOSS_IMPROVEMENT_EPSILON = 1e-6
_MAX_W_NORM = 3.0
_MIN_UNIQUE_RESPONSES = 3


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
            lo, hi = -np.inf, thresholds[0]
        elif yi == K:
            lo, hi = thresholds[-1], np.inf
        else:
            lo, hi = thresholds[yi - 2], thresholds[yi - 1]
        p = float(stats.norm.cdf((hi - scores[i]) / std) - stats.norm.cdf((lo - scores[i]) / std))
        total -= math.log(max(p, 1e-12))
    delta = w - prior_w
    total += 0.5 * float(l2) * float(delta @ delta)
    return float(total)


def _load_events(conn, question_id: str) -> List[Dict[str, Any]]:
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


def run_calibration_job(
    conn,
    job: Dict[str, Any],
    *,
    min_responses: int,
    l2: float = 2.0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Execute one calibration job. Updates the job row in-place and returns a result dict.
    Quality gates:
      - n >= min_responses
      - at least _MIN_UNIQUE_RESPONSES unique response values
      - optimizer convergence
      - loss_improvement > _LOSS_IMPROVEMENT_EPSILON
      - norm(new_w) <= _MAX_W_NORM
      - all values finite
    """
    job_id = int(job["id"])
    question_id = str(job["question_id"])
    now = int(time.time())
    attempt = int(job["attempt_count"]) + 1

    db.update_calibration_job(
        conn,
        job_id=job_id,
        status="running",
        started_at=now,
        attempt_count=attempt,
    )

    def _fail(reason: str, error: str = "") -> Dict[str, Any]:
        db.update_calibration_job(
            conn, job_id=job_id, status="failed",
            finished_at=int(time.time()), reason=reason, last_error=error or reason,
        )
        return {"job_id": job_id, "question_id": question_id, "outcome": "failed", "reason": reason}

    def _skip(reason: str, gates: Dict[str, Any]) -> Dict[str, Any]:
        db.update_calibration_job(
            conn, job_id=job_id, status="skipped",
            finished_at=int(time.time()), reason=reason, quality_gates=gates,
        )
        return {"job_id": job_id, "question_id": question_id, "outcome": "skipped", "reason": reason, "quality_gates": gates}

    active = db.get_active_question_parameter_version(conn, question_id)
    if active is None:
        return _fail("no_active_parameter_version")

    events = _load_events(conn, question_id)
    n = len(events)
    if n < min_responses:
        return _skip(
            f"insufficient_responses ({n} < {min_responses})",
            {"n_responses": n, "min_responses": min_responses},
        )

    unique_responses = len({e["response"] for e in events})
    if unique_responses < _MIN_UNIQUE_RESPONSES:
        return _skip(
            f"insufficient_response_diversity ({unique_responses} unique values)",
            {"unique_responses": unique_responses, "min_unique": _MIN_UNIQUE_RESPONSES},
        )

    theta = np.asarray([e["mu"] for e in events], dtype=np.float64)
    y = np.asarray([e["response"] for e in events], dtype=np.int64)
    prior_w = np.asarray(active["w"], dtype=np.float64).reshape(-1)
    thresholds = np.asarray(active["thresholds"], dtype=np.float64).reshape(-1)
    noise_var = float(active["noise_var"])

    if theta.shape[1] != prior_w.size:
        return _fail(f"dimension_mismatch: theta={theta.shape[1]} w={prior_w.size}")

    old_loss = _neg_loglik(
        prior_w, theta=theta, y=y, thresholds=thresholds,
        noise_var=noise_var, prior_w=prior_w, l2=l2,
    )

    try:
        result = optimize.minimize(
            lambda x: _neg_loglik(
                np.asarray(x, dtype=np.float64), theta=theta, y=y, thresholds=thresholds,
                noise_var=noise_var, prior_w=prior_w, l2=l2,
            ),
            prior_w,
            method="BFGS",
        )
    except Exception as exc:
        return _fail("optimizer_exception", str(exc)[:500])

    if not result.success:
        return _fail(f"optimizer_failed: {str(result.message)[:200]}")

    new_w = np.asarray(result.x, dtype=np.float64).reshape(-1)
    new_loss = _neg_loglik(
        new_w, theta=theta, y=y, thresholds=thresholds,
        noise_var=noise_var, prior_w=prior_w, l2=l2,
    )
    improvement = float(old_loss - new_loss)
    w_norm = float(np.linalg.norm(new_w))
    all_finite = bool(np.all(np.isfinite(new_w)))
    cosine_sim = float(
        np.dot(prior_w, new_w) / (np.linalg.norm(prior_w) * w_norm + 1e-12)
        if w_norm > 0 and np.linalg.norm(prior_w) > 0 else 0.0
    )
    delta_norm = float(np.linalg.norm(new_w - prior_w))

    gates = {
        "n_responses": n,
        "unique_responses": unique_responses,
        "old_loss": float(old_loss),
        "new_loss": float(new_loss),
        "loss_improvement": improvement,
        "improvement_passes": improvement > _LOSS_IMPROVEMENT_EPSILON,
        "w_norm": w_norm,
        "w_norm_passes": w_norm <= _MAX_W_NORM,
        "all_finite": all_finite,
        "cosine_similarity_to_prior": cosine_sim,
        "delta_norm": delta_norm,
    }

    passes = improvement > _LOSS_IMPROVEMENT_EPSILON and w_norm <= _MAX_W_NORM and all_finite

    if not passes:
        failing = [k for k in ("improvement_passes", "w_norm_passes", "all_finite") if not gates[k]]
        return _skip(f"gates_failed: {', '.join(failing)}", gates)

    # All gates passed.
    promoted_version_id: Optional[int] = None
    if not dry_run:
        promoted_version_id = db.insert_question_parameter_version(
            conn,
            question_id=question_id,
            w=new_w.tolist(),
            noise_var=noise_var,
            thresholds=thresholds.tolist(),
            source="estimated",
            estimation_method=METHOD,
            n_responses_used=n,
            performance_summary={
                "old_loss": float(old_loss),
                "new_loss": float(new_loss),
                "improvement": improvement,
                "w_norm": w_norm,
                "n_responses": n,
            },
            active=True,
        )
        db.insert_question_calibration_run(
            conn,
            question_id=question_id,
            status="promoted",
            n_responses=n,
            old_version=int(active["version"]),
            new_version=db.get_active_question_parameter_version(conn, question_id)["version"],
            active_promoted=True,
            method=METHOD,
            diagnostics=gates,
        )
        db.update_calibration_job(
            conn,
            job_id=job_id,
            status="succeeded",
            finished_at=int(time.time()),
            reason="gates_passed",
            old_loss=float(old_loss),
            new_loss=float(new_loss),
            loss_improvement=improvement,
            old_w=prior_w.tolist(),
            new_w=new_w.tolist(),
            quality_gates=gates,
            promoted_parameter_version_id=promoted_version_id,
        )
    else:
        db.update_calibration_job(
            conn,
            job_id=job_id,
            status="skipped",
            finished_at=int(time.time()),
            reason="dry_run",
            old_loss=float(old_loss),
            new_loss=float(new_loss),
            loss_improvement=improvement,
            old_w=prior_w.tolist(),
            new_w=new_w.tolist(),
            quality_gates=gates,
        )

    return {
        "job_id": job_id,
        "question_id": question_id,
        "outcome": "would_promote" if dry_run else "promoted",
        "promoted_version_id": promoted_version_id,
        "quality_gates": gates,
    }


def run_pending_calibrations(
    db_path: str,
    *,
    limit: Optional[int] = None,
    question_id: Optional[str] = None,
    min_responses: int = 50,
    dry_run: bool = False,
    force: bool = False,
) -> List[Dict[str, Any]]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        if force and question_id:
            # Force-queue the job even if one already exists.
            active_pv = db.get_active_question_parameter_version(conn, question_id)
            current_version_id = int(active_pv["id"]) if active_pv is not None else None
            conn.execute(
                """
                INSERT INTO calibration_jobs
                  (question_id, current_parameter_version_id, status, reason, created_at)
                VALUES (?, ?, 'pending', 'force_queued', ?);
                """,
                (question_id, current_version_id, int(time.time())),
            )
            conn.commit()

        jobs = db.list_calibration_jobs(conn, status="pending", question_id=question_id, limit=limit)
        if not jobs:
            print("No pending calibration jobs.")
            return []

        results = []
        for job in jobs:
            try:
                r = run_calibration_job(conn, job, min_responses=min_responses, dry_run=dry_run)
                results.append(r)
                outcome = r.get("outcome", "?")
                print(f"[{r['question_id']}] job={r['job_id']} outcome={outcome}")
            except Exception as exc:
                print(f"[{job['question_id']}] UNEXPECTED ERROR: {exc}", file=sys.stderr)
                try:
                    db.update_calibration_job(
                        conn, job_id=int(job["id"]), status="failed",
                        finished_at=int(time.time()), last_error=str(exc)[:500],
                    )
                except Exception:
                    pass
                results.append({"job_id": job["id"], "question_id": job["question_id"], "outcome": "error", "error": str(exc)})
        return results
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pending question calibration jobs")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--min-responses", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-queue and run even if job exists")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    results = run_pending_calibrations(
        args.db,
        limit=args.limit,
        question_id=args.question_id,
        min_responses=args.min_responses,
        dry_run=args.dry_run,
        force=args.force,
    )
    if args.as_json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
