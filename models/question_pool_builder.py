"""
Build inference pools from seed items plus optional LLM-generated questions with assigned loadings ``w``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from models.question_assignment import assign_w_to_candidates, cosine_similarity_matrix, embed_texts
from models.question_generation import generate_candidate_questions, validate_candidate_question


@dataclass
class GeneratedPoolResult:
    """Returned by build_generated_pool. Contains accepted questions plus full candidate metadata."""
    accepted: List[dict] = field(default_factory=list)
    all_candidates_metadata: List[dict] = field(default_factory=list)

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


def _dedupe_generated_candidates_with_meta(
    candidates: List[dict],
    seed_items: List[dict],
) -> Tuple[List[dict], List[int], List[dict]]:
    """
    Drop candidates too similar to any seed, then near-duplicates among generated.
    If fewer than ``_MIN_CANDIDATES`` remain but enough seed-safe rows exist, pad by
    preferring items least similar to seeds (relaxing pairwise rules only if needed).

    Returns:
        kept_candidates: Candidates that survived deduplication.
        kept_indices: Original indices of kept candidates within ``candidates``.
        per_original_meta: One metadata dict per original candidate (order-preserving).
            Keys: candidate_index, text, max_seed_similarity, max_kept_similarity, dedupe_failed.
    """
    if not candidates:
        return [], [], []

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
        meta = [
            {
                "candidate_index": i,
                "text": str(candidates[i].get("text", "")),
                "max_seed_similarity": float(max_seed_sim[i]),
                "max_kept_similarity": None,
                "dedupe_failed": True,
            }
            for i in range(len(candidates))
        ]
        return [], [], meta

    kept_idx = _greedy_dedupe_generated(seed_safe_idx, cand_emb, _SIM_THRESHOLD)

    if len(kept_idx) < _MIN_CANDIDATES and len(seed_safe_idx) >= _MIN_CANDIDATES:
        kept_idx = _fill_to_min_candidates(
            kept_idx,
            seed_safe_idx,
            cand_emb,
            max_seed_sim,
            _MIN_CANDIDATES,
            _SIM_THRESHOLD,
        )

    kept_set = set(kept_idx)

    # Compute max_kept_similarity for seed-safe non-kept candidates (retrospective against final kept set).
    if kept_idx:
        kept_cand_emb = cand_emb[np.array(kept_idx, dtype=np.int64)]
        sim_to_kept = cosine_similarity_matrix(cand_emb, kept_cand_emb)
        max_kept_sim_arr = np.max(sim_to_kept, axis=1)
    else:
        max_kept_sim_arr = np.zeros(len(candidates))

    meta = []
    for i in range(len(candidates)):
        seed_rejected = max_seed_sim[i] > _SIM_THRESHOLD
        is_kept = i in kept_set
        meta.append({
            "candidate_index": i,
            "text": str(candidates[i].get("text", "")),
            "max_seed_similarity": float(max_seed_sim[i]),
            # None for seed-rejected candidates (never reached dedupe) and for accepted ones.
            "max_kept_similarity": (
                None if seed_rejected or is_kept
                else float(max_kept_sim_arr[i])
            ),
            "dedupe_failed": not is_kept,
        })

    return [candidates[i] for i in kept_idx], kept_idx, meta


def build_generated_pool(
    llm_client,
    seed_items: List[dict],
    n_candidates: int = 10,
    k: int = 3,
    prompt: Optional[str] = None,
) -> GeneratedPoolResult:
    """
    Generate candidate question texts with the LLM client, deduplicate, assign provisional
    ``w`` vectors, and validate.

    Returns a ``GeneratedPoolResult`` with:
    - ``.accepted``: Normalized accepted generated-question dicts (id, text, w, source, …).
      This has the same structure as the old list return value.
    - ``.all_candidates_metadata``: One dict per raw LLM candidate (including rejected ones).

    Args:
        llm_client: Object with ``complete(prompt: str) -> str``.
        seed_items: Seeds with ``text`` and ``w`` (and optional ``id``).
        n_candidates: Number of LLM candidates to request.
        k: Nearest neighbors for ``w`` averaging in ``assign_w_to_candidates``.
        prompt: Pre-rendered prompt string from a PromptPolicy. If None, falls back to
                ``build_generation_prompt`` (backward-compatible). Passed through to
                ``generate_candidate_questions``.
    """
    from models.question_bank import DEFAULT_NOISE_VAR, DEFAULT_THRESHOLDS

    candidates = generate_candidate_questions(
        llm_client,
        seed_questions=seed_items,
        n_candidates=n_candidates,
        prompt=prompt,
    )
    if not candidates:
        return GeneratedPoolResult(accepted=[], all_candidates_metadata=[])

    deduped, kept_indices, all_meta = _dedupe_generated_candidates_with_meta(candidates, seed_items)

    # Initialize validation/acceptance fields for every candidate (filled in below for deduped ones).
    for m in all_meta:
        m["validation_passed"] = False
        m["validation_failure_reason"] = None
        m["accepted_into_pool"] = False
        m["question_id"] = None
        m["w"] = None
        m["noise_var"] = None
        m["thresholds"] = None
        m["nn_seed_ids"] = None
        m["nn_similarities"] = None

    if not deduped:
        return GeneratedPoolResult(accepted=[], all_candidates_metadata=all_meta)

    assigned = assign_w_to_candidates(deduped, seed_items, k=k)

    # Safety / quality gate: drop any candidate that fails text or loading-vector checks.
    valid = []
    for j, row in enumerate(assigned):
        orig_idx = kept_indices[j]
        m = all_meta[orig_idx]
        ok, reason = validate_candidate_question(row)
        if ok:
            m["validation_passed"] = True
            m["accepted_into_pool"] = True
            m["question_id"] = str(row["id"])
            w_arr = np.asarray(row["w"], dtype=np.float64).reshape(-1)
            m["w"] = w_arr.tolist()
            m["noise_var"] = float(row.get("noise_var", DEFAULT_NOISE_VAR))
            thr_raw = row.get("thresholds")
            m["thresholds"] = (
                np.asarray(thr_raw, dtype=np.float64).tolist()
                if thr_raw is not None
                else DEFAULT_THRESHOLDS.tolist()
            )
            m["nn_seed_ids"] = list(row.get("nn_seed_ids") or [])
            m["nn_similarities"] = list(row.get("nn_similarities") or [])
            valid.append(row)
        else:
            m["validation_failure_reason"] = reason
            import warnings
            warnings.warn(
                f"Generated question rejected ({reason}): {str(row.get('text', ''))[:80]}",
                stacklevel=2,
            )

    accepted = [_normalize_question_dict(row) for row in valid]
    return GeneratedPoolResult(accepted=accepted, all_candidates_metadata=all_meta)


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
