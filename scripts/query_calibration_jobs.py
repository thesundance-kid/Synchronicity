#!/usr/bin/env python3
"""Read-only view of calibration_jobs and question_calibration_runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db

DEFAULT_DB = str(PROJECT_ROOT / "data" / "pilot.db")


def query_calibration_jobs(db_path: str, *, status: str | None = None, question_id: str | None = None) -> list:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        jobs = db.list_calibration_jobs(conn, status=status, question_id=question_id)
        for job in jobs:
            pid = job.get("current_parameter_version_id")
            qid = job["question_id"]
            n = conn.execute(
                "SELECT COUNT(*) FROM question_performance_events WHERE question_id = ?;",
                (qid,),
            ).fetchone()[0]
            job["current_n_responses"] = int(n)
            active = db.get_active_question_parameter_version(conn, qid)
            job["active_parameter_version"] = int(active["version"]) if active else None
        return jobs
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect calibration job queue (read-only)")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--status", default=None, choices=["pending", "running", "succeeded", "failed", "skipped"])
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    jobs = query_calibration_jobs(args.db, status=args.status, question_id=args.question_id)
    if args.as_json:
        print(json.dumps(jobs, indent=2))
        return

    if not jobs:
        print("No calibration jobs found.")
        return

    counts: dict = {}
    for job in jobs:
        counts[job["status"]] = counts.get(job["status"], 0) + 1

    print(f"Calibration jobs: {len(jobs)} total — " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    print()
    for job in jobs:
        promoted = job.get("promoted_parameter_version_id")
        improvement = job.get("loss_improvement")
        print(
            f"  [{job['status']:10s}] job={job['id']:4d}  q={job['question_id']:10s}"
            f"  n_responses={job['current_n_responses']:4d}"
            f"  active_v={job['active_parameter_version']}"
            f"  improvement={improvement:.4f}" if improvement is not None else
            f"  [{job['status']:10s}] job={job['id']:4d}  q={job['question_id']:10s}"
            f"  n_responses={job['current_n_responses']:4d}"
            f"  active_v={job['active_parameter_version']}"
            f"  improvement=n/a"
        )
        if job.get("reason"):
            print(f"             reason: {job['reason']}")
        if promoted:
            print(f"             promoted→v{promoted}")


if __name__ == "__main__":
    main()
