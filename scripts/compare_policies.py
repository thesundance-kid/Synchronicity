#!/usr/bin/env python3
"""
Compare question-selection policies: random, fixed-order, EIG-adaptive.

Run from project root: python scripts/compare_policies.py

Runs a modest number of episodes per policy with synthetic users and prints
aggregate metrics by step and final-step averages, plus a short interpretation.
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.question_bank import load_questions
from models.evaluation import (
    run_multiple_episodes,
    average_l2_error_by_step,
    average_entropy_by_step,
    average_cov_trace_by_step,
    final_step_averages,
)


def main() -> None:
    questions_path = os.path.join(_project_root, "data", "questions.json")
    if not os.path.isfile(questions_path):
        print("Error: data/questions.json not found at", questions_path)
        sys.exit(1)

    questions = load_questions(questions_path, expected_dim=5)
    dim = 5
    n_episodes = 30
    max_questions = None  # use all questions per episode

    policies = ["random", "fixed_order", "eig"]
    results = {}

    for policy in policies:
        episodes = run_multiple_episodes(
            policy_name=policy,
            questions=questions,
            dim=dim,
            n_episodes=n_episodes,
            rng_seed=42,
            max_questions=max_questions,
        )
        results[policy] = episodes

    # Number of steps (all episodes should have same length)
    steps = sorted(set(s for ep in results["eig"] for h in ep["history"] for s in [h["step"]]))
    if not steps:
        print("No history recorded; check run_episode.")
        sys.exit(1)
    n_steps = max(steps)

    print("=" * 60)
    print("Policy comparison ({} episodes, {} questions per episode)".format(n_episodes, n_steps))
    print("=" * 60)

    # By-step averages (compact)
    print("\n--- Average L2 error by step ---")
    print("step  " + "  ".join("{:>10}".format(p) for p in policies))
    for step in range(1, n_steps + 1):
        row = []
        for p in policies:
            by_step = average_l2_error_by_step(results[p])
            row.append(by_step.get(step, float("nan")))
        print("{:4}  ".format(step) + "  ".join("{:10.4f}".format(x) for x in row))

    print("\n--- Average entropy by step ---")
    print("step  " + "  ".join("{:>10}".format(p) for p in policies))
    for step in range(1, n_steps + 1):
        row = []
        for p in policies:
            by_step = average_entropy_by_step(results[p])
            row.append(by_step.get(step, float("nan")))
        print("{:4}  ".format(step) + "  ".join("{:10.4f}".format(x) for x in row))

    print("\n--- Average cov_trace by step ---")
    print("step  " + "  ".join("{:>10}".format(p) for p in policies))
    for step in range(1, n_steps + 1):
        row = []
        for p in policies:
            by_step = average_cov_trace_by_step(results[p])
            row.append(by_step.get(step, float("nan")))
        print("{:4}  ".format(step) + "  ".join("{:10.4f}".format(x) for x in row))

    # Final-step averages
    print("\n" + "=" * 60)
    print("Final-step averages (after last question)")
    print("=" * 60)
    for p in policies:
        fin = final_step_averages(results[p])
        print("  {}:  L2_error = {:.4f},  entropy = {:.4f},  cov_trace = {:.4f}".format(
            p, fin["l2_error"], fin["entropy"], fin["cov_trace"]))

    # Interpretation
    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)
    feig = final_step_averages(results["eig"])
    frand = final_step_averages(results["random"])
    ffix = final_step_averages(results["fixed_order"])
    print("EIG-adaptive aims to reduce uncertainty (entropy) and estimation error (L2) faster")
    print("than random or fixed-order by asking the most informative question next.")
    if feig["l2_error"] < frand["l2_error"] and feig["entropy"] < frand["entropy"]:
        print("In this run, EIG achieves lower final L2 error and entropy than random,")
        print("consistent with adaptive selection helping.")
    elif feig["l2_error"] < frand["l2_error"] or feig["entropy"] < frand["entropy"]:
        print("In this run, EIG improves at least one of L2 error or entropy vs random.")
    else:
        print("In this run, EIG did not improve final metrics over random; try more episodes")
        print("or different seeds to see typical advantage.")
    print()


if __name__ == "__main__":
    main()
