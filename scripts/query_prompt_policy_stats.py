#!/usr/bin/env python3
"""
Read-only analysis script: prompt/probe-generation policy statistics.

Aggregates by prompt_policy_version_id and reports:
- number of generation requests
- number of candidates generated, accepted, dedupe-failed, validation-failed, selected
- acceptance rate, selection rate
- mean predicted EIG for selected questions (via question_performance_events)
- mean realized information gain for selected questions
- mean entropy drop for selected questions

Usage:
  python scripts/query_prompt_policy_stats.py [--db PATH] [--policy-id N] [--json]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _ro_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def query_prompt_policy_stats(
    db_path: str,
    policy_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Return one stats dict per prompt_policy_versions row.
    Opens the DB read-only; never mutates.
    """
    conn = _ro_conn(db_path)
    try:
        # Check that Phase 5 tables exist; return empty list on older DBs.
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        if "prompt_policy_versions" not in tables:
            return []

        where_clause = "WHERE ppv.id = ?" if policy_id is not None else ""
        params = (policy_id,) if policy_id is not None else ()

        rows = conn.execute(
            f"""
            SELECT ppv.id, ppv.name, ppv.version, ppv.strategy_type,
                   ppv.conditioning_mode, ppv.active
            FROM prompt_policy_versions ppv
            {where_clause}
            ORDER BY ppv.name ASC, ppv.version ASC;
            """,
            params,
        ).fetchall()

        results = []
        for ppv in rows:
            pid = int(ppv["id"])

            # Generation request stats.
            req_row = conn.execute(
                """
                SELECT COUNT(*) AS n_requests,
                       SUM(n_requested) AS n_requested_total,
                       SUM(n_returned) AS n_returned_total
                FROM llm_generation_requests
                WHERE prompt_policy_version_id = ?;
                """,
                (pid,),
            ).fetchone()
            n_requests = int(req_row["n_requests"] or 0)
            n_returned_total = int(req_row["n_returned_total"] or 0)

            # Candidate stats via generated_question_candidates.
            cand_row = conn.execute(
                """
                SELECT COUNT(*) AS n_logged,
                       SUM(accepted_into_pool)                                  AS n_accepted,
                       SUM(dedupe_failed)                                       AS n_dedupe_failed,
                       SUM(CASE WHEN validation_passed=0 AND dedupe_failed=0
                                THEN 1 ELSE 0 END)                              AS n_validation_failed,
                       SUM(CASE WHEN selected_at_step IS NOT NULL
                                THEN 1 ELSE 0 END)                              AS n_selected
                FROM generated_question_candidates
                WHERE prompt_policy_version_id = ?;
                """,
                (pid,),
            ).fetchone()
            n_logged = int(cand_row["n_logged"] or 0)
            n_accepted = int(cand_row["n_accepted"] or 0)
            n_dedupe_failed = int(cand_row["n_dedupe_failed"] or 0)
            n_validation_failed = int(cand_row["n_validation_failed"] or 0)
            n_selected = int(cand_row["n_selected"] or 0)

            acceptance_rate = n_accepted / n_logged if n_logged > 0 else None
            selection_rate = n_selected / n_accepted if n_accepted > 0 else None
            dedupe_failure_rate = n_dedupe_failed / n_logged if n_logged > 0 else None
            validation_failure_rate = n_validation_failed / n_logged if n_logged > 0 else None

            # Performance stats via join: candidates → question_performance_events.
            # Matches on (question_id, session_id) to scope to this policy's generated questions.
            has_qpe = "question_performance_events" in tables
            mean_predicted_eig = None
            mean_realized_ig = None
            mean_entropy_drop = None
            n_with_perf = 0

            if has_qpe:
                perf_row = conn.execute(
                    """
                    SELECT COUNT(qpe.id)               AS n_perf,
                           AVG(qpe.predicted_eig)      AS mean_predicted_eig,
                           AVG(qpe.realized_information_gain) AS mean_realized_ig,
                           AVG(qpe.entropy_before - qpe.entropy_after) AS mean_entropy_drop
                    FROM generated_question_candidates gqc
                    JOIN question_performance_events qpe
                      ON qpe.question_id = gqc.question_id
                     AND qpe.session_id  = gqc.session_id
                    WHERE gqc.prompt_policy_version_id = ?
                      AND gqc.selected_at_step IS NOT NULL;
                    """,
                    (pid,),
                ).fetchone()
                n_with_perf = int(perf_row["n_perf"] or 0)
                if n_with_perf > 0:
                    mean_predicted_eig = (
                        float(perf_row["mean_predicted_eig"])
                        if perf_row["mean_predicted_eig"] is not None
                        else None
                    )
                    mean_realized_ig = (
                        float(perf_row["mean_realized_ig"])
                        if perf_row["mean_realized_ig"] is not None
                        else None
                    )
                    mean_entropy_drop = (
                        float(perf_row["mean_entropy_drop"])
                        if perf_row["mean_entropy_drop"] is not None
                        else None
                    )

            results.append({
                "policy_id": pid,
                "name": ppv["name"],
                "version": int(ppv["version"]),
                "strategy_type": ppv["strategy_type"],
                "conditioning_mode": ppv["conditioning_mode"],
                "active": bool(ppv["active"]),
                "n_requests": n_requests,
                "n_candidates_logged": n_logged,
                "n_candidates_returned_by_llm": n_returned_total,
                "n_accepted": n_accepted,
                "n_dedupe_failed": n_dedupe_failed,
                "n_validation_failed": n_validation_failed,
                "n_selected": n_selected,
                "acceptance_rate": acceptance_rate,
                "selection_rate": selection_rate,
                "dedupe_failure_rate": dedupe_failure_rate,
                "validation_failure_rate": validation_failure_rate,
                "n_with_performance_data": n_with_perf,
                "mean_predicted_eig": mean_predicted_eig,
                "mean_realized_ig": mean_realized_ig,
                "mean_entropy_drop": mean_entropy_drop,
            })

        return results

    finally:
        conn.close()


def print_prompt_policy_stats(
    db_path: str,
    policy_id: Optional[int] = None,
) -> None:
    stats = query_prompt_policy_stats(db_path, policy_id=policy_id)
    if not stats:
        print("No prompt policy versions found.")
        return

    for s in stats:
        active_tag = " [ACTIVE]" if s["active"] else ""
        print(f"\n── Policy {s['policy_id']}: {s['name']} v{s['version']}{active_tag} ──")
        print(f"   strategy_type    : {s['strategy_type']}")
        print(f"   conditioning_mode: {s['conditioning_mode']}")
        print(f"   generation requests      : {s['n_requests']}")
        print(f"   candidates returned (LLM): {s['n_candidates_returned_by_llm']}")
        print(f"   candidates logged        : {s['n_candidates_logged']}")
        print(f"   accepted into pool       : {s['n_accepted']}"
              + (f"  ({s['acceptance_rate']:.1%})" if s['acceptance_rate'] is not None else ""))
        print(f"   dedupe-failed            : {s['n_dedupe_failed']}"
              + (f"  ({s['dedupe_failure_rate']:.1%})" if s['dedupe_failure_rate'] is not None else ""))
        print(f"   validation-failed        : {s['n_validation_failed']}"
              + (f"  ({s['validation_failure_rate']:.1%})" if s['validation_failure_rate'] is not None else ""))
        print(f"   selected (EIG asked)     : {s['n_selected']}"
              + (f"  ({s['selection_rate']:.1%} of accepted)" if s['selection_rate'] is not None else ""))
        if s["n_with_performance_data"] > 0:
            print(f"   perf data (n={s['n_with_performance_data']})")
            if s["mean_predicted_eig"] is not None:
                print(f"     mean predicted EIG : {s['mean_predicted_eig']:.4f}")
            if s["mean_realized_ig"] is not None:
                print(f"     mean realized IG   : {s['mean_realized_ig']:.4f}")
            if s["mean_entropy_drop"] is not None:
                print(f"     mean entropy drop  : {s['mean_entropy_drop']:.4f}")
        else:
            print("   perf data              : none yet")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt policy generation statistics (read-only).")
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "pilot.db"),
        help="Path to SQLite database (default: data/pilot.db)",
    )
    parser.add_argument(
        "--policy-id",
        type=int,
        default=None,
        help="Filter to a specific prompt_policy_versions.id",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text",
    )
    args = parser.parse_args()

    if args.json:
        stats = query_prompt_policy_stats(args.db, policy_id=args.policy_id)
        print(json.dumps(stats, indent=2))
    else:
        print_prompt_policy_stats(args.db, policy_id=args.policy_id)


if __name__ == "__main__":
    main()
