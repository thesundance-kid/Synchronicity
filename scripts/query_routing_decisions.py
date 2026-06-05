#!/usr/bin/env python3
"""Read-only view of recent policy routing decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db

DEFAULT_DB = str(PROJECT_ROOT / "data" / "pilot.db")


def query_routing_decisions(db_path: str, *, limit: int = 50, policy_id: int | None = None) -> list:
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        decisions = db.list_policy_routing_decisions(conn, prompt_policy_version_id=policy_id, limit=limit)
        policy_names: dict = {}
        for d in decisions:
            pid = int(d["prompt_policy_version_id"])
            if pid not in policy_names:
                row = conn.execute(
                    "SELECT name, version FROM prompt_policy_versions WHERE id = ?;", (pid,)
                ).fetchone()
                policy_names[pid] = f"{row['name']} v{row['version']}" if row else str(pid)
            d["policy_name"] = policy_names[pid]
        return decisions
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect policy routing decisions (read-only)")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--policy-id", type=int, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    decisions = query_routing_decisions(args.db, limit=args.limit, policy_id=args.policy_id)
    if args.as_json:
        print(json.dumps(decisions, indent=2))
        return

    if not decisions:
        print("No routing decisions found.")
        return

    counts: dict = {}
    for d in decisions:
        dt = d["decision_type"]
        counts[dt] = counts.get(dt, 0) + 1

    print(f"Last {len(decisions)} routing decisions:")
    print("  " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    print()
    print(f"{'Session':22}  {'Policy':30}  {'Decision':12}  {'ε':>5}  {'N eligible':>10}")
    print("-" * 85)
    for d in decisions:
        eps_str = f"{d['epsilon']:.2f}" if d["epsilon"] is not None else "n/a"
        print(
            f"{d['session_id']:22}  {d['policy_name']:30}  {d['decision_type']:12}  "
            f"{eps_str:>5}  {d['n_eligible_policies']:>10}"
        )


if __name__ == "__main__":
    main()
