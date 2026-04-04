"""
Assign provisional loading vectors w to generated questions via nearest seed neighbors.

Uses cosine similarity in embedding space; optional ``sentence-transformers`` if installed,
otherwise a simple deterministic hashing embedding (weaker semantics, no extra deps).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

# Default hashing dimension when sentence-transformers is unavailable
_HASH_DIM = 256

_ST_MODEL = None  # lazy SentenceTransformer instance


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed each string; return a matrix of shape ``(n_texts, embedding_dim)``.

    Tries ``sentence_transformers`` first; falls back to a fixed-size hashing embedding.
    """
    global _ST_MODEL

    if not texts:
        return np.zeros((0, _HASH_DIM), dtype=np.float64)

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        if _ST_MODEL is None:
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        vecs = _ST_MODEL.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float64)
    except Exception:
        return _embed_texts_hashing(texts)


def _embed_texts_hashing(texts: Sequence[str]) -> np.ndarray:
    """Deterministic bag-of-tokens hashing into ``_HASH_DIM`` dimensions (L2-normalized rows)."""
    n = len(texts)
    out = np.zeros((n, _HASH_DIM), dtype=np.float64)
    for i, t in enumerate(texts):
        toks = _tokenize_simple(t)
        for tok in toks:
            h = hash(tok) % _HASH_DIM
            out[i, h] += 1.0
        if out[i].sum() <= 0:
            out[i, 0] = 1.0
    # L2-normalize rows for cosine geometry
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return out / norms


def _tokenize_simple(text: str) -> List[str]:
    s = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [w for w in s.split() if w]


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity between rows of ``A`` and rows of ``B``.

    Args:
        A: shape ``(n, d)``
        B: shape ``(m, d)``

    Returns:
        Matrix of shape ``(n, m)`` with cosine similarity of ``A[i]`` with ``B[j]``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays.")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Embedding dim mismatch: A has {A.shape[1]}, B has {B.shape[1]}.")
    a_norm = np.linalg.norm(A, axis=1, keepdims=True)
    b_norm = np.linalg.norm(B, axis=1, keepdims=True)
    a_norm = np.maximum(a_norm, 1e-12)
    b_norm = np.maximum(b_norm, 1e-12)
    A_n = A / a_norm
    B_n = B / b_norm
    return A_n @ B_n.T


def assign_w_nearest_neighbors(
    candidate_text: str,
    seed_items: List[dict],
    seed_embeddings: np.ndarray,
    k: int = 3,
) -> Dict[str, Any]:
    """
    Embed the candidate, find top-``k`` seeds by cosine similarity, average their ``w``.

    ``seed_items[i]`` must include ``"w"`` (1D array-like). ``"id"`` is optional; index used if missing.

    Returns:
        ``{"w": np.ndarray, "nn_seed_ids": [...], "nn_similarities": [...]}``
    """
    if not seed_items:
        raise ValueError("seed_items must be non-empty.")
    if seed_embeddings.shape[0] != len(seed_items):
        raise ValueError("seed_embeddings rows must match len(seed_items).")
    k = min(int(k), len(seed_items))
    if k < 1:
        raise ValueError("k must be at least 1.")

    cand = embed_texts([candidate_text])
    sims = cosine_similarity_matrix(cand, seed_embeddings).reshape(-1)
    top_idx = np.argsort(-sims)[:k]
    top_sims = sims[top_idx]

    weights = np.maximum(top_sims.astype(np.float64), 0.0)
    if weights.sum() < 1e-12:
        weights = np.ones(k, dtype=np.float64) / k
    else:
        weights = weights / weights.sum()

    d = None
    w_sum = None
    nn_ids: List[Any] = []
    for j, idx in enumerate(top_idx):
        w_i = np.asarray(seed_items[int(idx)]["w"], dtype=np.float64).reshape(-1)
        if d is None:
            d = w_i.shape[0]
            w_sum = np.zeros(d, dtype=np.float64)
        elif w_i.shape[0] != d:
            raise ValueError(f"Seed w length mismatch at index {idx}.")
        w_sum = w_sum + weights[j] * w_i
        sid = seed_items[int(idx)].get("id", f"seed_{int(idx)}")
        nn_ids.append(sid)

    assert w_sum is not None
    return {
        "w": w_sum,
        "nn_seed_ids": nn_ids,
        "nn_similarities": top_sims.astype(float).tolist(),
    }


def assign_w_to_candidates(
    candidates: List[dict],
    seed_items: List[dict],
    k: int = 3,
) -> List[dict]:
    """
    Embed all seed texts once, then assign ``w`` to each candidate by nearest neighbors.

    Each output item includes ``"id"`` (``gen_<index>``), ``"w"``, ``"source": "generated"``,
    plus original candidate keys (e.g. ``"text"``).
    """
    seed_texts = [str(s.get("text", "")) for s in seed_items]
    seed_embeddings = embed_texts(seed_texts)

    out: List[dict] = []
    for idx, cand in enumerate(candidates):
        text = str(cand.get("text", ""))
        assignment = assign_w_nearest_neighbors(
            text,
            seed_items,
            seed_embeddings,
            k=k,
        )
        row = dict(cand)
        row["id"] = f"gen_{idx}"
        row["w"] = assignment["w"]
        row["source"] = "generated"
        out.append(row)
    return out
