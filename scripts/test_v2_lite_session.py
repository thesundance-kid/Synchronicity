#!/usr/bin/env python3
"""
Debug script: V2-lite pipeline (seed bank → generate → assign w → inference pool)
+ synthetic user + adaptive EIG loop. No FastAPI / DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.personality_state import PersonalityState
from models.question_bank import DEFAULT_THRESHOLDS, Question, load_question_pools_v2
from models.question_generation import DummyLLMClient
from models.question_pool_builder import build_generated_pool, build_session_inference_pool
from models.question_selection import select_next_question_eig
from models.simulator import sample_likert_response, sample_theta_true


def _dict_to_question(d: dict) -> Question:
    w = np.asarray(d["w"], dtype=np.float64).reshape(-1)
    thr = d.get("thresholds")
    thresholds = (
        np.asarray(thr, dtype=np.float64).reshape(-1)
        if thr is not None
        else DEFAULT_THRESHOLDS.copy()
    )
    return Question(
        id=str(d["id"]),
        text=str(d["text"]),
        w=w,
        noise_var=float(d.get("noise_var", 1.0)),
        thresholds=thresholds,
    )


def main() -> None:
    rng = np.random.default_rng(42)
    dim = 5
    questions_path = PROJECT_ROOT / "data" / "questions_v2.json"

    inference_seed, _heldout_pool = load_question_pools_v2(str(questions_path), expected_dim=dim)
    # Smaller seed subset keeps EIG scoring tractable for local debugging (full pool is very slow).
    inference_seed = inference_seed[:12]

    # Seeds for generation / pool (w as ndarray for NN assignment)
    seeds_for_gen = [{"id": q.id, "text": q.text, "w": q.w} for q in inference_seed]
    seed_stored = [
        {
            "id": q.id,
            "text": q.text,
            "w": q.w.astype(float).tolist(),
            "noise_var": float(q.noise_var),
            "thresholds": q.thresholds.astype(float).tolist(),
        }
        for q in inference_seed
    ]

    print("=== V2-lite debug session ===\n")
    print(f"Loaded {len(inference_seed)} seed inference questions from {questions_path.name}")

    generated = build_generated_pool(
        DummyLLMClient(),
        seeds_for_gen,
        n_candidates=8,
        k=3,
    )
    print(f"After dedupe + assign_w: {len(generated)} generated items")

    generated_ids = {d["id"] for d in generated}
    pool_dicts = build_session_inference_pool(seed_stored, generated if generated else None)
    questions = [_dict_to_question(d) for d in pool_dicts]
    print(f"Full inference pool size: {len(questions)} (seeds + generated)\n")

    theta_true = sample_theta_true(dim, rng)
    state = PersonalityState(dim=dim)
    asked: set[str] = set()

    n_steps = int(rng.integers(5, 9))  # 5–8 inclusive
    print(f"Adaptive EIG steps: {n_steps}\n")

    print(
        f"{'step':<5} {'question_id':<12} {'generated?':<12} {'response':<10} "
        f"{'eig':<10} {'entropy':<12}"
    )
    print("-" * 70)

    for step in range(n_steps):
        best, eig, _ = select_next_question_eig(state, questions, asked_ids=asked)
        if best is None:
            print(f"{step:<5} (no candidate left)")
            break
        asked.add(best.id)
        y = sample_likert_response(theta_true, best, rng)
        state.update_posterior_likert_laplace(
            w=best.w,
            y=y,
            thresholds=best.thresholds,
            noise_var=float(best.noise_var),
        )
        h_after = state.entropy()
        is_gen = best.id in generated_ids
        print(
            f"{step:<5} {best.id:<12} {str(is_gen):<12} {y:<10} {float(eig):<10.4f} {h_after:<12.4f}"
        )

    print("-" * 70)
    print(f"\nSelected question ids ({len(asked)}): {sorted(asked)}")
    gen_selected = sorted(asked & generated_ids)
    print(f"Generated ids selected ({len(gen_selected)}): {gen_selected}")
    print(f"Final entropy (nats): {state.entropy():.6f}")


if __name__ == "__main__":
    main()
