"""
Build inference pools from seed items plus optional LLM-generated questions with assigned loadings ``w``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from models.question_assignment import assign_w_to_candidates, cosine_similarity_matrix, embed_texts
from models.question_generation import generate_candidate_questions, validate_candidate_question

# Cosine similarity above this vs. a seed or another kept candidate counts as a near-duplicate.
_SIM_THRESHOLD = 0.9
# Minimum generated items to return when enough seed-safe candidates exist.
_MIN_CANDIDATES = 5


def _greedy_dedupe_generated(
    indices: Sequence[int],
    cand_emb: np.ndarray,
    threshold: float,
) -> List[int]:
    """Keep indices in given order; skip if too similar to any already kept."""
    kept: List[int] = []
    for i in indices:
        if not kept:
            kept.append(i)
            continue
        sims = cosine_similarity_matrix(cand_emb[[i]], cand_emb[np.array(kept, dtype=np.int64)])[0]
        if float(np.max(sims)) <= threshold:
            kept.append(i)
    return kept


def _fill_to_min_candidates(
    kept: List[int],
    seed_safe_idx: List[int],
    cand_emb: np.ndarray,
    max_seed_sim: np.ndarray,
    min_k: int,
    pair_threshold: float,
) -> List[int]:
    """
    Grow ``kept`` toward ``min_k`` using only seed-safe indices, preferring rows least
    similar to any seed. First try pairwise threshold; if still short, add remaining in
    that same order (relax near-duplicate among generated).
    """
    result = list(kept)
    if len(result) >= min_k:
        return result

    # Most distinct from seeds first
    ordered_rest = [i for i in sorted(seed_safe_idx, key=lambda j: float(max_seed_sim[j])) if i not in result]

    for i in ordered_rest:
        if len(result) >= min_k:
            break
        if not result:
            result.append(i)
            continue
        sims = cosine_similarity_matrix(cand_emb[[i]], cand_emb[np.array(result, dtype=np.int64)])[0]
        if float(np.max(sims)) <= pair_threshold:
            result.append(i)

    if len(result) < min_k:
        for i in ordered_rest:
            if len(result) >= min_k:
                break
            if i not in result:
                result.append(i)

    return result


def _dedupe_generated_candidates(
    candidates: List[dict],
    seed_items: List[dict],
) -> List[dict]:
    """
    Drop candidates too similar to any seed, then near-duplicates among generated.
    If fewer than ``_MIN_CANDIDATES`` remain but enough seed-safe rows exist, pad by
    preferring items least similar to seeds (relaxing pairwise rules only if needed).
    """
    if not candidates:
        return []

    seed_texts = [str(s.get("text", "")) for s in seed_items]
    cand_texts = [str(c.get("text", "")) for c in candidates]
    combined = embed_texts(seed_texts + cand_texts)
    n_seed = len(seed_texts)
    seed_emb = combined[:n_seed]
    cand_emb = combined[n_seed:]

    sim_to_seeds = cosine_similarity_matrix(cand_emb, seed_emb)
    max_seed_sim = np.max(sim_to_seeds, axis=1)

    seed_safe_idx = [i for i in range(len(candidates)) if max_seed_sim[i] <= _SIM_THRESHOLD]
    if not seed_safe_idx:
        return []

    kept_idx = _greedy_dedupe_generated(seed_safe_idx, cand_emb, _SIM_THRESHOLD)

    if len(kept_idx) >= _MIN_CANDIDATES:
        return [candidates[i] for i in kept_idx]

    if len(seed_safe_idx) < _MIN_CANDIDATES:
        return [candidates[i] for i in kept_idx]

    final_idx = _fill_to_min_candidates(
        kept_idx,
        seed_safe_idx,
        cand_emb,
        max_seed_sim,
        _MIN_CANDIDATES,
        _SIM_THRESHOLD,
    )
    return [candidates[i] for i in final_idx]


def build_generated_pool(
    llm_client,
    seed_items: List[dict],
    n_candidates: int = 10,
    k: int = 3,
) -> List[dict]:
    """
    Generate candidate question texts with the LLM client, then assign provisional ``w``
    vectors via nearest-neighbor weighting over ``seed_items``.

    Each returned dict includes at least ``id``, ``text``, and ``w`` (and ``source`` from
    the assignment step). ``w`` is a ``numpy.ndarray`` matching seed dimension.

    Args:
        llm_client: Object with ``complete(prompt: str) -> str``.
        seed_items: Seeds with ``text`` and ``w`` (and optional ``id``).
        n_candidates: Number of LLM candidates to request.
        k: Nearest neighbors for ``w`` averaging in ``assign_w_to_candidates``.
    """
    candidates = generate_candidate_questions(
        llm_client,
        seed_questions=seed_items,
        n_candidates=n_candidates,
    )
    if not candidates:
        return []

    candidates = _dedupe_generated_candidates(candidates, seed_items)
    if not candidates:
        return []

    assigned = assign_w_to_candidates(candidates, seed_items, k=k)

    # Safety / quality gate: drop any candidate that fails text or loading-vector checks.
    valid = []
    for row in assigned:
        ok, reason = validate_candidate_question(row)
        if ok:
            valid.append(row)
        else:
            import warnings
            warnings.warn(
                f"Generated question rejected ({reason}): {str(row.get('text', ''))[:80]}",
                stacklevel=2,
            )

    return [_normalize_question_dict(row) for row in valid]


def build_session_inference_pool(
    seed_items: List[dict],
    generated_items: Optional[List[dict]],
) -> List[dict]:
    """
    Combine seed and generated questions for a session inference pool.

    If ``generated_items`` is ``None``, returns ``seed_items`` only.
    Otherwise returns ``seed_items + generated_items`` (shallow concat of lists).
    """
    if generated_items is None:
        return list(seed_items)
    return list(seed_items) + list(generated_items)


def _normalize_question_dict(row: dict) -> dict:
    """Ensure required keys exist and ``w`` is a float64 numpy vector."""
    out = dict(row)
    out["id"] = str(out.get("id", ""))
    out["text"] = str(out.get("text", ""))
    w = out.get("w")
    if w is None:
        raise ValueError("Question dict missing required key 'w'.")
    out["w"] = np.asarray(w, dtype=np.float64).reshape(-1)
    return out
