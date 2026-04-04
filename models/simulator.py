"""
Synthetic user simulator for adaptive personality evaluation.

Generates true latent trait vectors and Likert responses consistent with
the project's ordinal Probit observation model:
  z = w^T theta_true + eps,  eps ~ N(0, noise_var)
  Likert category determined by thresholds (same as inference).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from models.question_bank import Question


def sample_theta_true(
    dim: int,
    rng: np.random.Generator,
    cov: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample a synthetic user's true latent trait vector.

    Args:
        dim: Dimension of the trait vector (e.g. 5 for Big Five).
        rng: NumPy random generator for reproducibility.
        cov: Covariance matrix for N(0, cov). If None, use 4*I to match the
            PersonalityState prior N(0, 4I). This ensures benchmark episodes
            are drawn from the same distribution the model assumes as its prior,
            so absolute error and entropy values are interpretable on a
            consistent scale.

    Returns:
        theta_true: shape (dim,) drawn from N(0, cov) or N(0, 4I).
    """
    if cov is None:
        cov = 4.0 * np.eye(dim, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (dim, dim):
        raise ValueError(f"cov must be ({dim}, {dim}), got {cov.shape}")
    # N(0, cov) via Cholesky: cov = L L^T, then theta = L @ z, z ~ N(0,I)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal(size=dim).astype(np.float64)
    return (L @ z).reshape(-1)


def sample_likert_response(
    theta_true: np.ndarray,
    question: "Question",
    rng: np.random.Generator,
) -> int:
    """
    Sample a single Likert response from the synthetic user for the given question.

    Uses the same observation model as inference:
      z = w^T theta_true + eps,  eps ~ N(0, question.noise_var)
      Response y in {1, ..., K} where category k corresponds to
      tau_{k-1} < z <= tau_k (tau_0 = -inf, tau_K = +inf).

    Args:
        theta_true: True latent trait vector, shape (d,).
        question: Question with w, noise_var, thresholds.
        rng: Random generator.

    Returns:
        y: Integer in {1, 2, ..., K} (K = len(thresholds)+1).
    """
    w = np.asarray(question.w, dtype=np.float64).reshape(-1)
    if w.shape[0] != theta_true.shape[0]:
        raise ValueError(
            "theta_true length {} does not match question w length {}.".format(
                theta_true.shape[0], w.shape[0]
            )
        )
    mean_z = float(w @ theta_true)
    std_z = float(np.sqrt(question.noise_var))
    z = mean_z + std_z * rng.standard_normal()
    thresholds = np.asarray(question.thresholds, dtype=np.float64).reshape(-1)
    K = thresholds.size + 1
    # Bin boundaries: (-inf, tau_0], (tau_0, tau_1], ..., (tau_{K-2}, tau_{K-1}], (tau_{K-1}, +inf)
    # Category k (1-indexed) when tau_{k-1} < z <= tau_k with tau_0=-inf, tau_K=+inf
    if z <= thresholds[0]:
        return 1
    for k in range(1, K - 1):
        if thresholds[k - 1] < z <= thresholds[k]:
            return k + 1
    return K
