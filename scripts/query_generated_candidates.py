#!/usr/bin/env python3
"""
Read-only analysis of generated_question_candidates.

Reports overall and per-session acceptance rates, dedupe/validation failure
rates, most common failure reasons, and which generated questions were
ultimately answered.

Usage:
  python scripts/query_generated_candidates.py
  python scripts/query_generated_candidates.py --db data/pilot.db
  python scripts/query_generated_candidates.py --session SESSION_ID
  python scripts/query_generated_candidates.py --top-reasons 10
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _open_ro(path: str) -> sqlite3.Connection:
    p = Path(path).resolve()
    if not p.exists():
        print(f"error: DB not found: {path}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _pct(num: int, denom: int) -> str:
    return f"{num/denom*100:.1f}%" if denom else "N/A"


def generated_candidates_stats(
    conn: sqlite3.Connection,
    *,
    session_id: Optional[str] = None,
    top_reasons: int = 5,
) -> Dict[str, Any]:
    """
    Return aggregated stats from generated_question_candidates.

    Keys:
      total, n_accepted, n_dedupe_failed, n_validation_failed, n_selected
      top_failure_reasons  [{reason, count}]
      per_session          [{session_id, total, n_accepted, n_selected}]
      accepted_texts       [{text, question_id, n_selected_across_sessions}]
    """
    where = "WHERE session_id = ?" if session_id else ""
    # When a session filter is active, additional conditions use AND instead of WHERE.
    where_and = "WHERE session_id = ? AND" if session_id else "WHERE"
    params = (session_id,) if session_id else ()

    try:
        agg = conn.execute(
            f"""
            SELECT
                COUNT(*)                                           AS total,
                SUM(accepted_into_pool)                           AS n_accepted,
                SUM(dedupe_failed)                                AS n_dedupe_failed,
                SUM(CASE WHEN dedupe_failed=0 AND validation_passed=0
                          THEN 1 ELSE 0 END)                      AS n_validation_failed,
                SUM(CASE WHEN selected_at_step IS NOT NULL
                          THEN 1 ELSE 0 END)                      AS n_selected
            FROM generated_question_candidates
            {where}
            """,
            params,
        ).fetchone()
    except sqlite3.OperationalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    top_reasons_rows = conn.execute(
        f"""
        SELECT validation_failure_reason, COUNT(*) AS cnt
        FROM generated_question_candidates
        {where_and} validation_failure_reason IS NOT NULL
        GROUP BY validation_failure_reason
        ORDER BY cnt DESC
        LIMIT ?
        """,
        (*params, top_reasons),
    ).fetchall()

    per_session_rows = conn.execute(
        f"""
        SELECT
            session_id,
            COUNT(*)                                       AS total,
            SUM(accepted_into_pool)                       AS n_accepted,
            SUM(CASE WHEN selected_at_step IS NOT NULL
                      THEN 1 ELSE 0 END)                  AS n_selected
        FROM generated_question_candidates
        {where}
        GROUP BY session_id
        ORDER BY MIN(created_at)
        """,
        params,
    ).fetchall()

    accepted_texts_rows = conn.execute(
        f"""
        SELECT text, question_id,
               SUM(CASE WHEN selected_at_step IS NOT NULL THEN 1 ELSE 0 END) AS n_selected
        FROM generated_question_candidates
        {where_and} accepted_into_pool = 1
        GROUP BY question_id
        ORDER BY n_selected DESC, text
        """,
        params,
    ).fetchall()

    return {
        "total": int(agg["total"] or 0),
        "n_accepted": int(agg["n_accepted"] or 0),
        "n_dedupe_failed": int(agg["n_dedupe_failed"] or 0),
        "n_validation_failed": int(agg["n_validation_failed"] or 0),
        "n_selected": int(agg["n_selected"] or 0),
        "top_failure_reasons": [
            {"reason": r["validation_failure_reason"], "count": r["cnt"]}
            for r in top_reasons_rows
        ],
        "per_session": [
            {
                "session_id": r["session_id"],
                "total": int(r["total"]),
                "n_accepted": int(r["n_accepted"] or 0),
                "n_selected": int(r["n_selected"] or 0),
            }
            for r in per_session_rows
        ],
        "accepted_texts": [
            {
                "text": r["text"],
                "question_id": r["question_id"],
                "n_selected": int(r["n_selected"] or 0),
            }
            for r in accepted_texts_rows
        ],
    }


def print_generated_candidates_stats(
    stats: Dict[str, Any],
    *,
    session_filter: Optional[str] = None,
    show_texts: bool = True,
) -> None:
    W = 72
    total = stats["total"]
    n_acc = stats["n_accepted"]
    n_ded = stats["n_dedupe_failed"]
    n_val = stats["n_validation_failed"]
    n_sel = stats["n_selected"]

    scope = f"session {session_filter[:12]}…" if session_filter else f"{len(stats['per_session'])} sessions"
    print(f"\n{'=' * W}")
    print(f"Generated Question Candidates  ({total} total across {scope})")
    print(f"{'=' * W}")

    if total == 0:
        print("\n  No generated candidates found (all sessions may be seed_only arm).")
        print(f"\n{'=' * W}")
        return

    col = 28
    print(f"\n  {'Accepted into pool:':<{col}} {n_acc:>4}  ({_pct(n_acc, total)})")
    print(f"  {'Dedupe failures:':<{col}} {n_ded:>4}  ({_pct(n_ded, total)})")
    print(f"  {'Validation failures:':<{col}} {n_val:>4}  ({_pct(n_val, total)})")
    print(f"  {'Selected (answered):':<{col}} {n_sel:>4}  ({_pct(n_sel, n_acc)} of accepted)")

    reasons = stats["top_failure_reasons"]
    if reasons:
        print(f"\n  Top validation_failure_reason values:")
        for r in reasons:
            print(f"    {r['count']:>4}  {r['reason']}")

    sessions = stats["per_session"]
    if sessions and not session_filter:
        print(f"\n  Per-session breakdown:")
        hdr = f"  {'session_id':<22}  {'total':>5}  {'accepted':>8}  {'selected':>8}"
        print(hdr)
        print(f"  {'-'*22}  {'-'*5}  {'-'*8}  {'-'*8}")
        for s in sessions:
            sid = s["session_id"][:20]
            print(f"  {sid:<22}  {s['total']:>5}  {s['n_accepted']:>8}  {s['n_selected']:>8}")

    if show_texts and stats["accepted_texts"]:
        print(f"\n  Accepted question texts ({len(stats['accepted_texts'])} unique):")
        for item in stats["accepted_texts"][:10]:
            sid_tag = f"[{item['question_id']}]" if item["question_id"] else "[no id]"
            sel_tag = f"  ← answered {item['n_selected']}x" if item["n_selected"] else ""
            snippet = item["text"]
            snippet = (snippet[:56] + "…") if len(snippet) > 57 else snippet
            print(f"    {sid_tag:<10}  {snippet}{sel_tag}")
        if len(stats["accepted_texts"]) > 10:
            print(f"    … {len(stats['accepted_texts']) - 10} more")

    print(f"\n{'=' * W}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect generated_question_candidates")
    parser.add_argument("--db", default=os.environ.get("PILOT_DB_PATH", "data/pilot.db"),
                        help="path to SQLite DB")
    parser.add_argument("--session", default=None, metavar="SESSION_ID",
                        help="filter to a single session")
    parser.add_argument("--top-reasons", type=int, default=5, metavar="N",
                        help="number of top failure reasons to show (default: 5)")
    parser.add_argument("--no-texts", action="store_true",
                        help="omit accepted question text listing")
    args = parser.parse_args()

    conn = _open_ro(args.db)
    stats = generated_candidates_stats(conn, session_id=args.session, top_reasons=args.top_reasons)
    print_generated_candidates_stats(stats, session_filter=args.session, show_texts=not args.no_texts)
    conn.close()


if __name__ == "__main__":
    main()
