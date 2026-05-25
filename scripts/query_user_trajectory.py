#!/usr/bin/env python3
"""
Read-only trajectory viewer for users and sessions.

--session-id SESSION_ID
    Prints every posterior_snapshot for that session, step by step.
    Shows entropy, delta-entropy, and the full mu vector (OCEAN traits).

--user-id USER_ID
    Shows the cross-session posterior history (user_posteriors) for all
    completed sessions, then the in-progress snapshots for the latest session.

Usage:
  python scripts/query_user_trajectory.py --session-id abc123def456
  python scripts/query_user_trajectory.py --user-id xyz789
  python scripts/query_user_trajectory.py --user-id xyz789 --db data/pilot.db
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIT_LABELS = ["O", "C", "E", "A", "N"]


def _open_ro(path: str) -> sqlite3.Connection:
    p = Path(path).resolve()
    if not p.exists():
        print(f"error: DB not found: {path}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _fmt_ts(ts: Optional[int]) -> str:
    if ts is None:
        return "N/A"
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_mu(mu_json: str) -> str:
    try:
        mu = json.loads(mu_json)
        parts = [f"{label}={v:+.2f}" for label, v in zip(TRAIT_LABELS, mu)]
        return "  ".join(parts)
    except Exception:
        return mu_json


def _entropy_bar(entropy: float, max_entropy: float, width: int = 20) -> str:
    if max_entropy <= 0:
        return " " * width
    filled = round(entropy / max_entropy * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Session trajectory
# ---------------------------------------------------------------------------

def session_trajectory(conn: sqlite3.Connection, session_id: str) -> Dict[str, Any]:
    """Return snapshot list and session metadata for one session."""
    try:
        sess = conn.execute(
            """
            SELECT session_id, mode, status, step, max_inference_questions,
                   arm, user_id, prior_session_id, created_at, updated_at
            FROM sessions WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    if sess is None:
        print(f"error: session not found: {session_id}", file=sys.stderr)
        sys.exit(1)

    snapshots = conn.execute(
        """
        SELECT step_idx, entropy, mu_json, sigma_json, created_at
        FROM posterior_snapshots
        WHERE session_id = ?
        ORDER BY step_idx
        """,
        (session_id,),
    ).fetchall()

    return {
        "session": dict(sess),
        "snapshots": [dict(s) for s in snapshots],
    }


def print_session_trajectory(data: Dict[str, Any]) -> None:
    W = 72
    sess = data["session"]
    snaps = data["snapshots"]

    print(f"\n{'=' * W}")
    print(f"Session: {sess['session_id']}")
    print(f"{'=' * W}")
    print(f"  mode={sess['mode']}  arm={sess['arm']}  status={sess['status']}"
          f"  step={sess['step']}/{sess['max_inference_questions']}")
    if sess["user_id"]:
        print(f"  user_id={sess['user_id']}")
    if sess["prior_session_id"]:
        print(f"  warm-started from session {sess['prior_session_id']}")
    print(f"  created: {_fmt_ts(sess['created_at'])}")

    if not snaps:
        print("\n  No posterior snapshots recorded.")
        print(f"\n{'=' * W}")
        return

    max_h = max(s["entropy"] for s in snaps)
    print(f"\n  Posterior snapshots  ({len(snaps)} steps, max entropy {max_h:.4f} nats)")
    print(f"\n  {'step':>4}  {'entropy':>8}  {'Δ':>7}  bar{'':17}  mu [O  C  E  A  N]")
    print(f"  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*20}  {'-'*30}")

    prev_h: Optional[float] = None
    for s in snaps:
        h = s["entropy"]
        delta = f"{h - prev_h:+.4f}" if prev_h is not None else "  prior"
        bar = _entropy_bar(h, max_h)
        mu_str = _fmt_mu(s["mu_json"])
        print(f"  {s['step_idx']:>4}  {h:>8.4f}  {delta:>7}  {bar}  {mu_str}")
        prev_h = h

    total_drop = snaps[0]["entropy"] - snaps[-1]["entropy"] if len(snaps) > 1 else 0.0
    print(f"\n  Total entropy reduction: {total_drop:.4f} nats"
          f"  ({total_drop/snaps[0]['entropy']*100:.1f}% of initial)")
    print(f"\n{'=' * W}")


# ---------------------------------------------------------------------------
# User trajectory
# ---------------------------------------------------------------------------

def user_trajectory(conn: sqlite3.Connection, user_id: str) -> Dict[str, Any]:
    """Return cross-session history and latest in-progress snapshots for a user."""
    try:
        user = conn.execute(
            "SELECT user_id, created_at FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    if user is None:
        print(f"error: user not found: {user_id}", file=sys.stderr)
        sys.exit(1)

    current = conn.execute(
        """
        SELECT latest_session_id, latest_step_idx, entropy, mu_json, updated_at
        FROM user_current_state WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()

    completed = conn.execute(
        """
        SELECT session_id, session_number, entropy, mu_json, created_at
        FROM user_posteriors WHERE user_id = ?
        ORDER BY session_number
        """,
        (user_id,),
    ).fetchall()

    # Snapshots for the latest session (in-progress or last completed).
    latest_snaps: List[Dict[str, Any]] = []
    latest_sess_id: Optional[str] = None
    if current:
        latest_sess_id = current["latest_session_id"]
        rows = conn.execute(
            """
            SELECT step_idx, entropy, mu_json
            FROM posterior_snapshots WHERE session_id = ?
            ORDER BY step_idx
            """,
            (latest_sess_id,),
        ).fetchall()
        latest_snaps = [dict(r) for r in rows]

    all_sessions = conn.execute(
        """
        SELECT session_id, status, step, max_inference_questions, arm, created_at
        FROM sessions WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,),
    ).fetchall()

    return {
        "user_id": user["user_id"],
        "user_created_at": user["created_at"],
        "current_state": dict(current) if current else None,
        "completed_sessions": [dict(r) for r in completed],
        "latest_session_id": latest_sess_id,
        "latest_snapshots": latest_snaps,
        "all_sessions": [dict(r) for r in all_sessions],
    }


def print_user_trajectory(data: Dict[str, Any]) -> None:
    W = 72
    print(f"\n{'=' * W}")
    print(f"User: {data['user_id']}")
    print(f"{'=' * W}")
    print(f"  registered: {_fmt_ts(data['user_created_at'])}")
    print(f"  sessions (all): {len(data['all_sessions'])}"
          f"  completed: {len(data['completed_sessions'])}")

    # Per-session summary table.
    all_s = data["all_sessions"]
    if all_s:
        print(f"\n  {'session_id':<22}  {'status':<10}  {'arm':<20}  {'step':>4}  created")
        print(f"  {'-'*22}  {'-'*10}  {'-'*20}  {'-'*4}  {'-'*20}")
        for s in all_s:
            sid = s["session_id"][:20]
            print(f"  {sid:<22}  {s['status']:<10}  {s['arm']:<20}"
                  f"  {s['step']:>4}  {_fmt_ts(s['created_at'])}")

    # Cross-session posterior history.
    completed = data["completed_sessions"]
    if completed:
        max_h = max(r["entropy"] for r in completed)
        print(f"\n  Completed-session posteriors  ({len(completed)} sessions):")
        print(f"\n  {'#':>3}  {'entropy':>8}  bar{'':17}  mu [O  C  E  A  N]")
        print(f"  {'-'*3}  {'-'*8}  {'-'*20}  {'-'*30}")
        for r in completed:
            bar = _entropy_bar(r["entropy"], max_h)
            mu_str = _fmt_mu(r["mu_json"])
            print(f"  {r['session_number']:>3}  {r['entropy']:>8.4f}  {bar}  {mu_str}")
    else:
        print("\n  No completed sessions yet.")

    # Current state.
    cur = data["current_state"]
    if cur:
        print(f"\n  Current state (after latest inference answer):")
        print(f"    session:  {cur['latest_session_id']}")
        print(f"    step:     {cur['latest_step_idx']}")
        print(f"    entropy:  {cur['entropy']:.4f} nats")
        print(f"    mu:       {_fmt_mu(cur['mu_json'])}")
        print(f"    updated:  {_fmt_ts(cur['updated_at'])}")

    # In-progress session snapshots.
    snaps = data["latest_snapshots"]
    if snaps:
        latest_sid = data["latest_session_id"]
        max_h = max(s["entropy"] for s in snaps)
        print(f"\n  Step-by-step snapshots for latest session ({latest_sid}):")
        print(f"\n  {'step':>4}  {'entropy':>8}  {'Δ':>7}  bar{'':17}  mu [O  C  E  A  N]")
        print(f"  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*20}  {'-'*30}")
        prev_h: Optional[float] = None
        for s in snaps:
            h = s["entropy"]
            delta = f"{h - prev_h:+.4f}" if prev_h is not None else "  prior"
            bar = _entropy_bar(h, max_h)
            mu_str = _fmt_mu(s["mu_json"])
            print(f"  {s['step_idx']:>4}  {h:>8.4f}  {delta:>7}  {bar}  {mu_str}")
            prev_h = h

    print(f"\n{'=' * W}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect posterior trajectories")
    parser.add_argument("--db", default=os.environ.get("PILOT_DB_PATH", "data/pilot.db"),
                        help="path to SQLite DB")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--session-id", metavar="SESSION_ID",
                     help="show step-by-step snapshots for one session")
    grp.add_argument("--user-id", metavar="USER_ID",
                     help="show cross-session and in-progress trajectory for a user")
    args = parser.parse_args()

    conn = _open_ro(args.db)

    if args.session_id:
        data = session_trajectory(conn, args.session_id)
        print_session_trajectory(data)
    else:
        data = user_trajectory(conn, args.user_id)
        print_user_trajectory(data)

    conn.close()


if __name__ == "__main__":
    main()
