#!/usr/bin/env python3
"""Read-only view of prompt policy scores, routing stats, and pending score jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db

DEFAULT_DB = str(PROJECT_ROOT / "data" / "pilot.db")


def query_policy_scores(db_path: str) -> list:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        policies = db.list_prompt_policy_versions(conn)
        results = []
        for p in policies:
            pid = int(p["id"])
            score_row = db.get_latest_policy_score(conn, pid)
            n_routed = db.count_routed_sessions_for_policy(conn, pid)
            pending_jobs = db.list_policy_score_jobs(conn, status="pending", prompt_policy_version_id=pid)
            results.append({
                "policy_id": pid,
                "name": p["name"],
                "version": p["version"],
                "active": p["active"],
                "routing_enabled": p["routing_enabled"],
                "n_routed_sessions": n_routed,
                "latest_reward_score": score_row["reward_score"] if score_row else None,
                "score_components": score_row["metrics"] if score_row else None,
                "score_computed_at": score_row["created_at"] if score_row else None,
                "pending_score_jobs": len(pending_jobs),
            })
        results.sort(key=lambda r: (r["latest_reward_score"] or float("-inf")), reverse=True)
        return results
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect prompt policy scores (read-only)")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    rows = query_policy_scores(args.db)
    if args.as_json:
        print(json.dumps(rows, indent=2))
        return

    if not rows:
        print("No policies found.")
        return

    print(f"{'ID':>4}  {'Name':25}  {'v':>3}  {'Active':6}  {'Routing':7}  {'Routed':6}  {'Score':>8}  {'Pending':>7}")
    print("-" * 80)
    for r in rows:
        score_str = f"{r['latest_reward_score']:.4f}" if r["latest_reward_score"] is not None else "n/a"
        print(
            f"{r['policy_id']:>4}  {r['name']:25}  {r['version']:>3}  "
            f"{'yes' if r['active'] else 'no':6}  "
            f"{'yes' if r['routing_enabled'] else 'no':7}  "
            f"{r['n_routed_sessions']:>6}  "
            f"{score_str:>8}  "
            f"{r['pending_score_jobs']:>7}"
        )
    print()
    print("Notes:")
    print("  Score = 0.30*acceptance + 0.20*selection + 0.35*realized_ig - penalties")
    print("  Run recompute_policy_scores.py to refresh pending score jobs")


if __name__ == "__main__":
    main()
