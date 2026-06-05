"""
Epsilon-greedy prompt policy routing.

Selects a routing-enabled policy for each session without touching the active=1
flag (which stays as the fallback/admin concept). Routing decisions are logged to
policy_routing_decisions for observability.

Decision types:
  'only_option'   — exactly one routing-enabled policy exists
  'under_tested'  — at least one policy has fewer than min_completed_sessions
  'exploration'   — epsilon draw: random policy selected
  'exploitation'  — best-scoring policy selected
  'fallback'      — no routing-enabled policies; fall back to the active policy
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

from app import db


_DEFAULT_STRATEGY = "epsilon_greedy"
_DEFAULT_EPSILON = 0.20
_DEFAULT_MIN_SESSIONS = 10


def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name, "").strip()
        return float(v) if v else default
    except (ValueError, TypeError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, "").strip()
        return int(v) if v else default
    except (ValueError, TypeError):
        return default


def select_prompt_policy_for_session(
    conn,
    *,
    session_id: str,
    rng: Optional[random.Random] = None,
    epsilon: Optional[float] = None,
    min_completed_sessions: Optional[int] = None,
    routing_strategy: str = _DEFAULT_STRATEGY,
) -> Optional[Dict[str, Any]]:
    """
    Choose a routing-enabled policy for this session using epsilon-greedy routing.

    Algorithm:
    1. Gather all routing-enabled policies (latest version per name).
    2. If none exist, fall back to the globally active policy (decision_type='fallback').
    3. If exactly one exists, always pick it ('only_option').
    4. If any policy has < min_completed_sessions routed sessions, choose uniformly
       among the under-tested subset ('under_tested').
    5. Otherwise, with probability epsilon pick a random policy ('exploration');
       with probability 1-epsilon pick the highest-scoring policy ('exploitation').
       If no reward scores exist, always explore ('exploration').

    Records a policy_routing_decisions row for every call.
    Returns the selected policy dict (same shape as _ppv_row_to_dict), or None.
    """
    _eps = epsilon if epsilon is not None else _env_float("PROMPT_POLICY_EPSILON", _DEFAULT_EPSILON)
    _min_sess = min_completed_sessions if min_completed_sessions is not None else _env_int(
        "PROMPT_POLICY_MIN_COMPLETED_SESSIONS", _DEFAULT_MIN_SESSIONS
    )
    _rng = rng or random

    eligible = db.get_routing_enabled_policies(conn)

    # Fallback: no routing-enabled policies.
    if not eligible:
        fallback = db.get_active_prompt_policy_version(conn)
        if fallback is not None:
            db.insert_policy_routing_decision(
                conn,
                session_id=session_id,
                prompt_policy_version_id=int(fallback["id"]),
                routing_strategy=routing_strategy,
                decision_type="fallback",
                epsilon=_eps,
                n_eligible_policies=0,
                scores_considered=None,
            )
        return fallback

    # Exactly one option.
    if len(eligible) == 1:
        chosen = eligible[0]
        db.insert_policy_routing_decision(
            conn,
            session_id=session_id,
            prompt_policy_version_id=int(chosen["id"]),
            routing_strategy=routing_strategy,
            decision_type="only_option",
            epsilon=_eps,
            n_eligible_policies=1,
            scores_considered=None,
        )
        return chosen

    # Build score + session-count map for all eligible policies.
    scored: List[Dict[str, Any]] = []
    for policy in eligible:
        pid = int(policy["id"])
        n_sessions = db.count_routed_sessions_for_policy(conn, pid)
        score_row = db.get_latest_policy_score(conn, pid)
        scored.append({
            "policy": policy,
            "policy_id": pid,
            "n_routed_sessions": n_sessions,
            "reward_score": score_row["reward_score"] if score_row is not None else None,
        })

    scores_log = [
        {
            "policy_id": s["policy_id"],
            "name": s["policy"]["name"],
            "n_routed_sessions": s["n_routed_sessions"],
            "reward_score": s["reward_score"],
        }
        for s in scored
    ]

    # Under-tested: any policy below minimum sessions threshold.
    under_tested = [s for s in scored if s["n_routed_sessions"] < _min_sess]
    if under_tested:
        chosen_s = _rng.choice(under_tested)
        decision_type = "under_tested"
    else:
        # Epsilon-greedy among all eligible.
        if _rng.random() < _eps:
            chosen_s = _rng.choice(scored)
            decision_type = "exploration"
        else:
            # Exploitation: best reward score. Policies with no score are treated
            # as having a very low score so they get picked before never-scored ones
            # only when exploration forces it.
            has_scores = [s for s in scored if s["reward_score"] is not None]
            if not has_scores:
                chosen_s = _rng.choice(scored)
                decision_type = "exploration"
            else:
                chosen_s = max(has_scores, key=lambda s: s["reward_score"])
                decision_type = "exploitation"

    chosen = chosen_s["policy"]
    db.insert_policy_routing_decision(
        conn,
        session_id=session_id,
        prompt_policy_version_id=int(chosen["id"]),
        routing_strategy=routing_strategy,
        decision_type=decision_type,
        epsilon=_eps,
        n_eligible_policies=len(eligible),
        scores_considered=scores_log,
    )
    return chosen
