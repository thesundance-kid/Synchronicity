#!/usr/bin/env python3
"""
Recompute prompt policy reward scores from accumulated lineage data.

Finds policy_score_jobs with status='pending', recomputes the reward score
using acceptance/selection/realized-IG lineage, and persists the result to
prompt_policy_scores.

Run AFTER run_pending_calibrations.py so that realized-IG in QPE reflects
calibrated w vectors before policy reward scores are refreshed.

Usage:
  python scripts/recompute_policy_scores.py
  python scripts/recompute_policy_scores.py --dry-run
  python scripts/recompute_policy_scores.py --policy-id 1 --force
  python scripts/recompute_policy_scores.py --limit 5 --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db
from scripts.query_prompt_policy_stats import query_prompt_policy_stats

DEFAULT_DB = str(PROJECT_ROOT / "data" / "pilot.db")


def _nz(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (ValueError, TypeError):
        return default


def _compute_reward(stats_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute weighted reward score from a query_prompt_policy_stats row.
    Returns dict with reward_score and components.
    """
    acceptance = _nz(stats_row.get("acceptance_rate"))
    selection = _nz(stats_row.get("selection_rate"))
    realized_ig = _nz(stats_row.get("mean_realized_ig"))
    dedupe_fail = _nz(stats_row.get("dedupe_failure_rate"))
    validation_fail = _nz(stats_row.get("validation_failure_rate"))
    n_requests_failed = _nz(stats_row.get("n_requests_failed"))
    n_requests = max(1.0, _nz(stats_row.get("n_requests"), 0.0))
    failure_rate = n_requests_failed / n_requests

    reward = (
        0.30 * acceptance
        + 0.20 * selection
        + 0.35 * realized_ig
        - 0.10 * dedupe_fail
        - 0.20 * validation_fail
        - 0.20 * failure_rate
    )
    components = {
        "acceptance_rate": acceptance,
        "selection_rate": selection,
        "mean_realized_ig": realized_ig,
        "dedupe_failure_rate": dedupe_fail,
        "validation_failure_rate": validation_fail,
        "request_failure_rate": failure_rate,
        "weights": {
            "acceptance": 0.30,
            "selection": 0.20,
            "realized_ig": 0.35,
            "dedupe_penalty": -0.10,
            "validation_penalty": -0.20,
            "failure_penalty": -0.20,
        },
    }
    return {"reward_score": float(reward), "components": components}


def run_policy_score_job(
    conn,
    job: Dict[str, Any],
    stats_by_policy_id: Dict[int, Dict[str, Any]],
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    job_id = int(job["id"])
    policy_id = int(job["prompt_policy_version_id"])
    now = int(time.time())
    attempt = int(job["attempt_count"]) + 1

    db.update_policy_score_job(
        conn,
        job_id=job_id,
        status="running",
        started_at=now,
        attempt_count=attempt,
    )

    stats_row = stats_by_policy_id.get(policy_id)
    if stats_row is None:
        db.update_policy_score_job(
            conn, job_id=job_id, status="skipped",
            finished_at=int(time.time()), reason="no_data_for_policy",
        )
        return {"job_id": job_id, "policy_id": policy_id, "outcome": "skipped", "reason": "no_data_for_policy"}

    try:
        scored = _compute_reward(stats_row)
    except Exception as exc:
        db.update_policy_score_job(
            conn, job_id=job_id, status="failed",
            finished_at=int(time.time()), last_error=str(exc)[:500],
        )
        return {"job_id": job_id, "policy_id": policy_id, "outcome": "failed", "error": str(exc)}

    old_score_row = db.get_latest_policy_score(conn, policy_id)
    old_score = old_score_row["reward_score"] if old_score_row else None

    if not dry_run:
        db.insert_prompt_policy_score(
            conn,
            prompt_policy_version_id=policy_id,
            n_requests=int(stats_row.get("n_requests") or 0),
            n_candidates=int(stats_row.get("n_accepted") or 0),
            n_selected=int(stats_row.get("n_selected") or 0),
            reward_score=scored["reward_score"],
            metrics={**scored["components"], "raw_stats": {
                k: stats_row[k] for k in stats_row if k not in ("prompt_template",)
            }},
        )
        db.update_policy_score_job(
            conn, job_id=job_id, status="succeeded",
            finished_at=int(time.time()),
            old_score=old_score,
            new_score=scored["reward_score"],
            score_components=scored["components"],
        )
    else:
        db.update_policy_score_job(
            conn, job_id=job_id, status="skipped",
            finished_at=int(time.time()), reason="dry_run",
            old_score=old_score,
            new_score=scored["reward_score"],
            score_components=scored["components"],
        )

    return {
        "job_id": job_id,
        "policy_id": policy_id,
        "outcome": "would_score" if dry_run else "scored",
        "old_score": old_score,
        "new_score": scored["reward_score"],
        "components": scored["components"],
    }


def recompute_policy_scores(
    db_path: str,
    *,
    limit: Optional[int] = None,
    policy_id: Optional[int] = None,
    dry_run: bool = False,
    force: bool = False,
) -> List[Dict[str, Any]]:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        if force and policy_id:
            conn.execute(
                """
                INSERT INTO policy_score_jobs
                  (prompt_policy_version_id, status, reason, created_at)
                VALUES (?, 'pending', 'force_queued', ?);
                """,
                (int(policy_id), int(time.time())),
            )
            conn.commit()

        jobs = db.list_policy_score_jobs(
            conn,
            status="pending",
            prompt_policy_version_id=policy_id,
            limit=limit,
        )
        if not jobs:
            print("No pending policy score jobs.")
            return []

        # Load all stats once — cheaper than per-job queries.
        all_stats = query_prompt_policy_stats(db_path)
        stats_by_policy_id = {int(r["policy_id"]): r for r in all_stats}

        results = []
        for job in jobs:
            try:
                r = run_policy_score_job(conn, job, stats_by_policy_id, dry_run=dry_run)
                results.append(r)
                outcome = r.get("outcome", "?")
                new_score = r.get("new_score")
                score_str = f"{new_score:.4f}" if new_score is not None else "n/a"
                print(f"[policy {r['policy_id']}] job={r['job_id']} outcome={outcome} score={score_str}")
            except Exception as exc:
                print(f"[policy {job['prompt_policy_version_id']}] UNEXPECTED ERROR: {exc}", file=sys.stderr)
                try:
                    db.update_policy_score_job(
                        conn, job_id=int(job["id"]), status="failed",
                        finished_at=int(time.time()), last_error=str(exc)[:500],
                    )
                except Exception:
                    pass
                results.append({"job_id": job["id"], "policy_id": job["prompt_policy_version_id"], "outcome": "error", "error": str(exc)})
        return results
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute prompt policy reward scores")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--policy-id", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-queue and run even if job exists")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    results = recompute_policy_scores(
        args.db,
        limit=args.limit,
        policy_id=args.policy_id,
        dry_run=args.dry_run,
        force=args.force,
    )
    if args.as_json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
