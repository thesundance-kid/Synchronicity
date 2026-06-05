#!/usr/bin/env python3
"""
Read-only script: inspect selection score logs for recent sessions.

Shows for each inference selection step:
  - session_id, step_idx
  - selected question (rank 0) and its score components
  - top-K alternative candidates and their scores
  - whether the winner clearly won (margin) or narrowly won
  - source: seed/generated
  - EIG and composite score components

Usage:
  python scripts/query_selection_logs.py [--limit N] [--session SESSION_ID]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db

DB_PATH = os.environ.get("PILOT_DB_PATH", str(PROJECT_ROOT / "data" / "pilot.db"))
NARROW_WIN_THRESHOLD = 0.05  # scores within this fraction of the winner are "narrow"


def _fmt(v, width=8, decimals=4) -> str:
    if v is None:
        return " " * width
    return f"{v:{width}.{decimals}f}"


def run(db_path: str, limit: int = 20, session_id_filter: str | None = None) -> None:
    if not Path(db_path).exists():
        print(f"DB not found: {db_path}")
        return

    conn = db.connect(db_path)

    # Fetch recent sessions
    if session_id_filter:
        rows = conn.execute(
            "SELECT session_id, created_at FROM sessions WHERE session_id = ? ORDER BY created_at DESC;",
            (session_id_filter,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT session_id, created_at FROM sessions ORDER BY created_at DESC LIMIT ?;",
            (limit,),
        ).fetchall()

    if not rows:
        print("No sessions found.")
        return

    for sess_row in rows:
        sid = sess_row["session_id"]
        logs = db.list_selection_score_logs(conn, sid)
        if not logs:
            continue

        print(f"\n{'='*80}")
        print(f"Session: {sid}  (created_at={sess_row['created_at']})")
        print(f"{'='*80}")

        # Group by step
        by_step: dict[int, list] = {}
        for log in logs:
            by_step.setdefault(log["step_idx"], []).append(log)

        for step_idx in sorted(by_step):
            step_logs = sorted(by_step[step_idx], key=lambda x: x["candidate_rank"])
            winner = next((l for l in step_logs if l["selected"]), None)
            alternatives = [l for l in step_logs if not l["selected"]]

            print(f"\n  Step {step_idx}:")
            if winner:
                win_score = winner["selection_score"]
                eig = winner["expected_information_gain"]
                src = winner["question_source"]
                cal = winner["calibration_status"] or "—"
                print(f"    WINNER  [{src:9s}] {winner['question_id']:20s}  "
                      f"score={_fmt(win_score, 7, 4)}  EIG={_fmt(eig, 7, 4)}  "
                      f"nov={_fmt(winner.get('semantic_novelty'), 6, 3)}  "
                      f"cal={cal}")

                if alternatives:
                    best_alt_score = max(a["selection_score"] for a in alternatives)
                    margin = win_score - best_alt_score
                    margin_pct = abs(margin / win_score) if win_score != 0 else 0.0
                    narrow = margin_pct < NARROW_WIN_THRESHOLD
                    print(f"    margin={margin:+.4f} ({'NARROW' if narrow else 'clear'})")

                for alt in alternatives:
                    a_score = alt["selection_score"]
                    a_eig = alt["expected_information_gain"]
                    a_src = alt["question_source"]
                    print(f"    alt[{alt['candidate_rank']}] [{a_src:9s}] {alt['question_id']:20s}  "
                          f"score={_fmt(a_score, 7, 4)}  EIG={_fmt(a_eig, 7, 4)}  "
                          f"nov={_fmt(alt.get('semantic_novelty'), 6, 3)}")
            else:
                print(f"    (no selected=1 row for step {step_idx})")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query selection score logs.")
    parser.add_argument("--limit", type=int, default=20, help="Max sessions to show")
    parser.add_argument("--session", type=str, default=None, help="Filter to one session_id")
    parser.add_argument("--db", type=str, default=DB_PATH, help="SQLite DB path")
    args = parser.parse_args()
    run(db_path=args.db, limit=args.limit, session_id_filter=args.session)


if __name__ == "__main__":
    main()
