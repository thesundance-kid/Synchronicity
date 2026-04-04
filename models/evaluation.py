"""
Policy evaluation for adaptive personality questioning.

Runs episodes with synthetic users and records step-by-step metrics.
Policies: random, fixed-order, EIG-adaptive.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from models.personality_state import PersonalityState
from models.question_bank import Question
from models.question_selection import select_next_question_eig
from models.simulator import sample_likert_response, sample_theta_true


def run_episode(
    policy_name: str,
    questions: Sequence[Question],
    dim: int,
    rng: np.random.Generator,
    max_questions: Optional[int] = None,
    fixed_order_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a single episode: one synthetic user, ask questions per policy, record metrics.

    Args:
        policy_name: One of "random", "fixed_order", "eig".
        questions: Full question list (e.g. from load_questions).
        dim: Latent dimension (must match question w length).
        rng: Random generator for theta_true and responses (and random policy).
        max_questions: Max number of questions to ask. If None, use len(questions).
        fixed_order_ids: For "fixed_order" policy, optional list of question ids
            defining the order. If None, use the order of the questions list.

    Returns:
        dict with:
          - theta_true: (dim,) true trait vector
          - policy: policy_name
          - history: list of per-step dicts with step, question_id, response,
            mu, cov_trace, entropy, l2_error
    """
    n_available = len(questions)
    if n_available == 0:
        return {"theta_true": np.zeros(dim), "policy": policy_name, "history": []}

    n_steps = min(max_questions, n_available) if max_questions is not None else n_available
    theta_true = sample_theta_true(dim, rng)
    state = PersonalityState(dim=dim)
    asked_ids: set = set()
    history: List[Dict[str, Any]] = []
    id_to_question = {q.id: q for q in questions}

    # Fixed-order: sequence of question ids (unique, preserve order)
    if policy_name == "fixed_order":
        if fixed_order_ids is not None:
            seen: set = set()
            order_ids = []
            for qid in fixed_order_ids:
                if qid in id_to_question and qid not in seen:
                    seen.add(qid)
                    order_ids.append(qid)
                    if len(order_ids) >= n_steps:
                        break
        else:
            order_ids = [q.id for q in questions][:n_steps]
    else:
        order_ids = []

    for step in range(1, n_steps + 1):
        # Select next question by policy
        if policy_name == "random":
            remaining = [q for q in questions if q.id not in asked_ids]
            if not remaining:
                break
            q = rng.choice(remaining)
        elif policy_name == "fixed_order":
            if step - 1 >= len(order_ids):
                break
            qid = order_ids[step - 1]
            q = id_to_question.get(qid)
            if q is None:
                break
        else:  # eig
            best, _, _ = select_next_question_eig(state, list(questions), asked_ids=asked_ids)
            if best is None:
                break
            q = best

        asked_ids.add(q.id)
        response = sample_likert_response(theta_true, q, rng)

        state.update_posterior_likert_laplace(
            w=q.w,
            y=response,
            thresholds=q.thresholds,
            noise_var=q.noise_var,
        )

        l2_error = float(np.linalg.norm(state.mu - theta_true))
        cov_trace = float(np.trace(state.sigma))
        ent = state.entropy()

        history.append({
            "step": step,
            "question_id": q.id,
            "response": response,
            "mu": state.mu.copy(),
            "cov_trace": cov_trace,
            "entropy": ent,
            "l2_error": l2_error,
        })

    return {
        "theta_true": theta_true,
        "policy": policy_name,
        "history": history,
    }


def run_multiple_episodes(
    policy_name: str,
    questions: Sequence[Question],
    dim: int,
    n_episodes: int,
    rng_seed: int = 0,
    max_questions: Optional[int] = None,
    fixed_order_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run multiple episodes with the same policy; each episode gets its own RNG state.

    Args:
        policy_name, questions, dim: Passed through to run_episode.
        n_episodes: Number of episodes to run.
        rng_seed: Seed for the root RNG; each episode uses rng_seed + episode index.
        max_questions, fixed_order_ids: Passed through to run_episode.

    Returns:
        List of episode dicts (same structure as run_episode return).
    """
    root = np.random.default_rng(rng_seed)
    episodes = []
    for i in range(n_episodes):
        rng = np.random.default_rng(root.integers(0, 2**31))
        ep = run_episode(
            policy_name=policy_name,
            questions=questions,
            dim=dim,
            rng=rng,
            max_questions=max_questions,
            fixed_order_ids=fixed_order_ids,
        )
        episodes.append(ep)
    return episodes


def average_l2_error_by_step(episodes: List[Dict[str, Any]]) -> Dict[int, float]:
    """
    Average L2 error (||mu - theta_true||) at each step across episodes.

    Returns:
        dict mapping step number (1-based) to mean l2_error over episodes
        that have that step.
    """
    from collections import defaultdict
    by_step: Dict[int, List[float]] = defaultdict(list)
    for ep in episodes:
        for h in ep["history"]:
            by_step[h["step"]].append(h["l2_error"])
    return {s: float(np.mean(vals)) for s, vals in by_step.items()}


def average_entropy_by_step(episodes: List[Dict[str, Any]]) -> Dict[int, float]:
    """
    Average posterior entropy at each step across episodes.
    """
    from collections import defaultdict
    by_step: Dict[int, List[float]] = defaultdict(list)
    for ep in episodes:
        for h in ep["history"]:
            by_step[h["step"]].append(h["entropy"])
    return {s: float(np.mean(vals)) for s, vals in by_step.items()}


def average_cov_trace_by_step(episodes: List[Dict[str, Any]]) -> Dict[int, float]:
    """
    Average trace(sigma) at each step across episodes.
    """
    from collections import defaultdict
    by_step: Dict[int, List[float]] = defaultdict(list)
    for ep in episodes:
        for h in ep["history"]:
            by_step[h["step"]].append(h["cov_trace"])
    return {s: float(np.mean(vals)) for s, vals in by_step.items()}


def final_step_averages(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Mean L2 error, entropy, and cov_trace at the final step of each episode.

    Episodes with empty history are skipped for the mean.
    """
    if not episodes:
        return {"l2_error": 0.0, "entropy": 0.0, "cov_trace": 0.0}
    finals = [ep["history"][-1] for ep in episodes if ep["history"]]
    if not finals:
        return {"l2_error": 0.0, "entropy": 0.0, "cov_trace": 0.0}
    return {
        "l2_error": float(np.mean([f["l2_error"] for f in finals])),
        "entropy": float(np.mean([f["entropy"] for f in finals])),
        "cov_trace": float(np.mean([f["cov_trace"] for f in finals])),
    }
