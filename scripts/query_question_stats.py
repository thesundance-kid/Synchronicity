#!/usr/bin/env python3
"""
Read-only analysis of question_performance_events.

Aggregates per-question stats from the pilot DB: how often each question was answered,
mean predicted EIG, mean realized information gain, response distribution, and which
parameter versions were active.

Usage:
  python scripts/query_question_stats.py
  python scripts/query_question_stats.py --db data/pilot.db
  python scripts/query_question_stats.py --sort mean_rig --min-n 3 --top 10
"""

from __future__ import annotations

import argparse
import json
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


def _load_question_texts(questions_path: str) -> Dict[str, str]:
    try:
        with open(questions_path, encoding="utf-8") as f:
            raw = json.load(f)
        items: list = []
        if isinstance(raw, dict):
            items.extend(raw.get("inference_pool", []))
            items.extend(raw.get("heldout_pool", []))
        else:
            items = list(raw)
        return {it["id"]: it["text"] for it in items if "id" in it and "text" in it}
    except Exception:
        return {}


def question_stats(
    conn: sqlite3.Connection,
    *,
    min_n: int = 1,
    sort: str = "n_answered",
) -> List[Dict[str, Any]]:
    """
    Return per-question aggregate stats from question_performance_events.

    Each dict has keys:
      question_id, source, n_answered, mean_predicted_eig, mean_realized_ig,
      mean_entropy_drop, response_dist {response_value: count},
      param_version_counts {str(version): count}
    """
    try:
        rows = conn.execute(
            """
            SELECT
                question_id,
                question_source,
                COUNT(*)                            AS n_answered,
                AVG(predicted_eig)                  AS mean_predicted_eig,
                AVG(realized_information_gain)      AS mean_realized_ig,
                AVG(entropy_before - entropy_after) AS mean_entropy_drop,
                GROUP_CONCAT(parameter_version)     AS pv_csv
            FROM question_performance_events
            GROUP BY question_id, question_source
            HAVING COUNT(*) >= ?
            """,
            (min_n,),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    resp_rows = conn.execute(
        """
        SELECT question_id, response_value, COUNT(*) AS cnt
        FROM question_performance_events
        GROUP BY question_id, response_value
        ORDER BY question_id, response_value
        """,
    ).fetchall()
    resp_dist: Dict[str, Dict[int, int]] = {}
    for r in resp_rows:
        resp_dist.setdefault(r["question_id"], {})[r["response_value"]] = r["cnt"]

    results: List[Dict[str, Any]] = []
    for row in rows:
        qid = row["question_id"]
        pv_counts: Dict[str, int] = {}
        for v in (row["pv_csv"] or "").split(","):
            v = v.strip()
            if v:
                pv_counts[v] = pv_counts.get(v, 0) + 1

        results.append({
            "question_id": qid,
            "source": row["question_source"],
            "n_answered": row["n_answered"],
            "mean_predicted_eig": row["mean_predicted_eig"],
            "mean_realized_ig": row["mean_realized_ig"],
            "mean_entropy_drop": row["mean_entropy_drop"],
            "response_dist": resp_dist.get(qid, {}),
            "param_version_counts": pv_counts,
        })

    sort_key = {
        "n_answered": lambda r: -r["n_answered"],
        "mean_eig": lambda r: -(r["mean_predicted_eig"] or 0.0),
        "mean_rig": lambda r: -(r["mean_realized_ig"] or 0.0),
    }.get(sort, lambda r: -r["n_answered"])
    results.sort(key=sort_key)
    return results


def print_question_stats(
    results: List[Dict[str, Any]],
    question_texts: Dict[str, str],
    *,
    top_n: Optional[int] = None,
) -> None:
    W = 72
    if not results:
        print("\nNo events found in question_performance_events.")
        return

    total_events = sum(r["n_answered"] for r in results)
    print(f"\n{'=' * W}")
    print(f"Question Performance  ({len(results)} questions, {total_events} events total)")
    print(f"{'=' * W}")

    shown = results[:top_n] if top_n else results
    for r in shown:
        qid = r["question_id"]
        text = question_texts.get(qid, "")
        snippet = (text[:64] + "…") if len(text) > 65 else text

        meig = f"{r['mean_predicted_eig']:.4f}" if r["mean_predicted_eig"] is not None else "N/A  "
        mrig = f"{r['mean_realized_ig']:.4f}"   if r["mean_realized_ig"]   is not None else "N/A  "
        drop = f"{r['mean_entropy_drop']:.4f}"  if r["mean_entropy_drop"]  is not None else "N/A  "

        print(f"\n  {qid:<14}  [{r['source']}]  n={r['n_answered']}")
        if snippet:
            print(f"  {snippet}")
        print(f"  pred_eig={meig}  realized_ig={mrig}  entropy_drop={drop}")

        rd = r["response_dist"]
        if rd:
            total = sum(rd.values())
            parts = [f"{v}:{rd[v]}({rd[v]/total*100:.0f}%)" for v in sorted(rd)]
            print(f"  responses: {' | '.join(parts)}")

        pvc = r["param_version_counts"]
        if pvc:
            pv_str = " ".join(f"v{k}:{cnt}" for k, cnt in sorted(pvc.items()))
            print(f"  param_versions: {pv_str}")

    if top_n and len(results) > top_n:
        print(f"\n  … {len(results) - top_n} more questions not shown (--top {top_n})")
    print(f"\n{'=' * W}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect question_performance_events")
    parser.add_argument("--db", default=os.environ.get("PILOT_DB_PATH", "data/pilot.db"),
                        help="path to SQLite DB (default: PILOT_DB_PATH or data/pilot.db)")
    parser.add_argument("--questions",
                        default=os.environ.get("QUESTIONS_PATH", "data/questions_v2.json"),
                        help="path to questions_v2.json for question text lookup")
    parser.add_argument("--sort", choices=["n_answered", "mean_eig", "mean_rig"],
                        default="n_answered")
    parser.add_argument("--min-n", type=int, default=1, metavar="N",
                        help="minimum answers to include a question (default: 1)")
    parser.add_argument("--top", type=int, default=None, metavar="N",
                        help="show only top N questions")
    args = parser.parse_args()

    conn = _open_ro(args.db)
    texts = _load_question_texts(args.questions)
    results = question_stats(conn, min_n=args.min_n, sort=args.sort)
    print_question_stats(results, texts, top_n=args.top)
    conn.close()


if __name__ == "__main__":
    main()
