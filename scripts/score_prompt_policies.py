#!/usr/bin/env python3
"""
Read-only prompt-policy reward scoring.

Uses the existing policy -> request -> candidate -> performance lineage to assign
a balanced reward score. This script does not write prompt_policy_scores; it is
the read-only analysis pass for choosing future policy allocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.query_prompt_policy_stats import query_prompt_policy_stats

DEFAULT_DB = str(PROJECT_ROOT / "data" / "pilot.db")


def _nz(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def score_prompt_policies(db_path: str) -> List[Dict[str, Any]]:
    rows = query_prompt_policy_stats(db_path)
    scored: List[Dict[str, Any]] = []
    for row in rows:
        acceptance = _nz(row.get("acceptance_rate"))
        selection = _nz(row.get("selection_rate"))
        realized_ig = _nz(row.get("mean_realized_ig"))
        dedupe_fail = _nz(row.get("dedupe_failure_rate"))
        validation_fail = _nz(row.get("validation_failure_rate"))
        failures = _nz(row.get("n_requests_failed"))
        requests = max(1.0, _nz(row.get("n_requests"), 0.0))
        failure_rate = failures / requests

        reward = (
            0.30 * acceptance
            + 0.20 * selection
            + 0.35 * realized_ig
            - 0.10 * dedupe_fail
            - 0.20 * validation_fail
            - 0.20 * failure_rate
        )
        out = dict(row)
        out["reward_score"] = float(reward)
        out["reward_components"] = {
            "acceptance_rate": acceptance,
            "selection_rate": selection,
            "mean_realized_ig": realized_ig,
            "dedupe_failure_rate": dedupe_fail,
            "validation_failure_rate": validation_fail,
            "request_failure_rate": failure_rate,
        }
        scored.append(out)
    scored.sort(key=lambda x: x["reward_score"], reverse=True)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Score prompt policies without mutating the DB")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = score_prompt_policies(args.db)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    for row in rows:
        print(
            "#{id} {name} v{version}: reward={reward:.4f}, accepted={accepted}, selected={selected}".format(
                id=row["policy_id"],
                name=row["name"],
                version=row["version"],
                reward=row["reward_score"],
                accepted=row.get("n_accepted", 0),
                selected=row.get("n_selected", 0),
            )
        )


if __name__ == "__main__":
    main()
