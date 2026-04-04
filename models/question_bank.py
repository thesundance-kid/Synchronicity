"""
Question bank: lightweight question representation and JSON loader.

Each question has an id, text, loading vector w (shape d), and optional
noise_var and thresholds for the ordinal Probit response model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Default 5-point Likert thresholds and noise variance used when not in JSON.
DEFAULT_THRESHOLDS = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
DEFAULT_NOISE_VAR = 1.0


@dataclass
class Question:
    """
    A single personality question with ordinal (Likert) response model parameters.

    Attributes:
        id: Unique string identifier (e.g. "o_01", "c_02").
        text: The question text shown to the user.
        w: Loading vector of shape (d,) mapping latent traits to this item.
        noise_var: Variance of Gaussian noise in latent score z = w^T theta + eps.
        thresholds: Strictly increasing (K-1,) cutpoints for K Likert categories.
    """

    id: str
    text: str
    w: np.ndarray
    noise_var: float = DEFAULT_NOISE_VAR
    thresholds: np.ndarray = field(default_factory=lambda: DEFAULT_THRESHOLDS.copy())

    def __post_init__(self) -> None:
        self.w = np.asarray(self.w, dtype=np.float64).reshape(-1)
        self.thresholds = np.asarray(self.thresholds, dtype=np.float64).reshape(-1)
        if self.w.ndim != 1 or self.w.size == 0:
            raise ValueError("w must be a non-empty 1D array.")
        if self.thresholds.ndim != 1 or self.thresholds.size == 0:
            raise ValueError("thresholds must be a non-empty 1D array.")
        if not np.all(np.diff(self.thresholds) > 0):
            raise ValueError("thresholds must be strictly increasing.")
        if self.noise_var <= 0:
            raise ValueError("noise_var must be positive.")


def load_questions(
    path: str,
    expected_dim: Optional[int] = None,
) -> List[Question]:
    """
    Load questions from a JSON file and optionally validate latent dimension.

    Each JSON item must have at least:
      - "id": str
      - "text": str
      - "w": list of d numbers

    Optional keys (defaults applied if missing):
      - "noise_var": float (default 1.0)
      - "thresholds": list of (K-1) numbers (default [-1.5, -0.5, 0.5, 1.5])

    Args:
        path: Path to the JSON file (e.g. "data/questions.json").
        expected_dim: If provided, every question's w must have this length.
            If None, the dimension is inferred from the first question and
            all others must match.

    Returns:
        List of Question instances with consistent w shape.

    Raises:
        ValueError: If any w has inconsistent length or validation fails.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Question bank not found: {path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("JSON root must be a list of question objects.")

    questions: List[Question] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} must be a dict, got {type(item)}.")
        qid = item.get("id")
        text = item.get("text")
        w_raw = item.get("w")
        if qid is None or text is None or w_raw is None:
            raise ValueError(f"Item {i}: missing required key among id, text, w.")
        w = np.asarray(w_raw, dtype=np.float64).reshape(-1)
        if expected_dim is not None and w.shape[0] != expected_dim:
            raise ValueError(
                f"Item {i} (id={qid}): w has length {w.shape[0]}, expected {expected_dim}."
            )
        if questions and w.shape[0] != questions[0].w.shape[0]:
            raise ValueError(
                f"Item {i} (id={qid}): w length {w.shape[0]} does not match "
                f"first question dimension {questions[0].w.shape[0]}."
            )
        noise_var = float(item.get("noise_var", DEFAULT_NOISE_VAR))
        thresholds_raw = item.get("thresholds")
        if thresholds_raw is not None:
            thresholds = np.asarray(thresholds_raw, dtype=np.float64).reshape(-1)
        else:
            thresholds = DEFAULT_THRESHOLDS.copy()
        questions.append(
            Question(id=str(qid), text=str(text), w=w, noise_var=noise_var, thresholds=thresholds)
        )

    return questions


def load_question_pools_v2(
    path: str,
    expected_dim: Optional[int] = None,
) -> Tuple[List[Question], List[Question]]:
    """
    Load a v2 question bank with separate inference and held-out pools.

    Expected JSON shape:
      {
        "schema_version": "v2",
        "inference_pool": [ {id, text, w, ...}, ... ],
        "heldout_pool":   [ {id, text, w, ...}, ... ]
      }

    Args:
        path: Path to the JSON file (e.g. "data/questions_v2.json").
        expected_dim: If provided, every question's w must have this length.
            If None, the dimension is inferred from the first question across
            both pools and all others must match.

    Returns:
        (inference_pool, heldout_pool)
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Question bank not found: {path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("v2 JSON root must be a dict.")
    if raw.get("schema_version") != "v2":
        raise ValueError('Expected schema_version "v2".')

    inference_raw = raw.get("inference_pool")
    heldout_raw = raw.get("heldout_pool")
    if not isinstance(inference_raw, list) or not isinstance(heldout_raw, list):
        raise ValueError("v2 JSON must include inference_pool and heldout_pool lists.")

    # Infer expected_dim if not provided.
    inferred_dim: Optional[int] = expected_dim
    for pool in (inference_raw, heldout_raw):
        for item in pool:
            if not isinstance(item, dict) or "w" not in item:
                continue
            w = np.asarray(item["w"], dtype=np.float64).reshape(-1)
            inferred_dim = int(w.shape[0])
            break
        if inferred_dim is not None:
            break

    def _parse_pool(items: List[Dict]) -> List[Question]:
        qs: List[Question] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Pool item {i} must be a dict, got {type(item)}.")
            qid = item.get("id")
            text = item.get("text")
            w_raw = item.get("w")
            if qid is None or text is None or w_raw is None:
                raise ValueError(f"Pool item {i}: missing required key among id, text, w.")

            w = np.asarray(w_raw, dtype=np.float64).reshape(-1)
            if inferred_dim is not None and w.shape[0] != inferred_dim:
                raise ValueError(
                    f"Pool item {i} (id={qid}): w has length {w.shape[0]}, expected {inferred_dim}."
                )

            noise_var = float(item.get("noise_var", DEFAULT_NOISE_VAR))
            thresholds_raw = item.get("thresholds")
            thresholds = (
                np.asarray(thresholds_raw, dtype=np.float64).reshape(-1)
                if thresholds_raw is not None
                else DEFAULT_THRESHOLDS.copy()
            )
            qs.append(Question(id=str(qid), text=str(text), w=w, noise_var=noise_var, thresholds=thresholds))
        return qs

    inference_pool = _parse_pool(inference_raw)
    heldout_pool = _parse_pool(heldout_raw)

    # Ensure IDs are unique across both pools.
    ids = [q.id for q in inference_pool] + [q.id for q in heldout_pool]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids found across pools in v2 question bank.")

    return inference_pool, heldout_pool
