"""
Personality State: Multidimensional Gaussian prior and Laplace-approximate
posterior updates for ordinal (Likert) responses.

We model our belief over a d-dimensional personality vector θ as a
multivariate normal N(mu, Sigma). The mean `mu` is our current best estimate;
the covariance matrix `Sigma` encodes uncertainty and trait correlations.

Covariance semantics:
  Sigma[i, i]  = variance of trait i (high = we are uncertain about that trait)
  Sigma[i, j]  > 0  implies traits i and j tend to move together
  Sigma[i, j]  < 0  implies inverse relationship
  Sigma[i, j] ≈ 0   implies near-independence

For Likert data we use an ordinal Probit model on a 1D latent score
    z = w^T theta + eps,  eps ~ N(0, noise_var)
and thresholds τ separating the ordered categories. Because this likelihood
is not conjugate with a Gaussian prior, we use a Laplace approximation:
  1) Find the MAP estimate θ̂ that maximizes the posterior
  2) Approximate the posterior as N(θ̂, H^{-1}) where H is the Hessian of the
     negative log-posterior at θ̂.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy import optimize, stats


class PersonalityState:
    """
    Tracks the posterior belief over a user's personality as a d-dimensional Gaussian.

    Attributes:
        mu: np.ndarray of shape (d,) - Mean vector for the latent traits.
        sigma: np.ndarray of shape (d, d) - Covariance matrix of the Gaussian belief.

    The covariance matrix `sigma`:
      - Diagonal entries sigma[i, i] = variance of trait i (high = uncertain)
      - Off-diagonal sigma[i, j]     = covariance between traits i and j
        (positive = move together, negative = move in opposite directions).
    """

    def __init__(
        self,
        mu_init: Optional[np.ndarray] = None,
        sigma_init: Optional[np.ndarray] = None,
        dim: Optional[int] = None,
    ) -> None:
        """
        Initialize the personality state with a Gaussian prior.

        Args:
            mu_init: Initial mean vector (d,). If provided, its length defines d.
            sigma_init: Initial covariance matrix (d, d). If provided, its size
                must match d inferred from mu_init (if any) or `dim`.
            dim: Number of latent traits. Used only when mu_init and sigma_init
                are both None. Defaults to 5 for backwards compatibility.

        Math Note:
            We use sigma_init = 4 * I (identity) so each trait starts with
            variance 4 (on a typical 1-5 scale), meaning we are maximally
            uncertain. A diffuse prior lets the data dominate; we start agnostic.
            The prior mean of 0 is arbitrary; you could use 3 (a neutral Likert
            value) in an alternative parameterization tied directly to item scores.
        """
        # Determine dimensionality d
        if mu_init is not None:
            mu_arr = np.asarray(mu_init, dtype=np.float64).reshape(-1)
            d = mu_arr.shape[0]
        elif sigma_init is not None:
            sigma_arr = np.asarray(sigma_init, dtype=np.float64)
            if sigma_arr.ndim != 2 or sigma_arr.shape[0] != sigma_arr.shape[1]:
                raise ValueError("sigma_init must be a square (d, d) matrix.")
            d = sigma_arr.shape[0]
            mu_arr = np.zeros(d, dtype=np.float64)
        else:
            # Default dimension when nothing is specified.
            if dim is None:
                dim = 5
            d = int(dim)
            if d <= 0:
                raise ValueError("dim must be a positive integer.")
            mu_arr = np.zeros(d, dtype=np.float64)

        if sigma_init is None:
            # Math Note: 4 * I gives each trait variance 4 for high initial uncertainty.
            sigma_arr = 4.0 * np.eye(d, dtype=np.float64)
        else:
            sigma_arr = np.asarray(sigma_init, dtype=np.float64)

        if sigma_arr.shape != (d, d):
            raise ValueError(f"sigma must be ({d}, {d}), got {sigma_arr.shape}")
        if not np.allclose(sigma_arr, sigma_arr.T):
            raise ValueError("sigma must be symmetric.")
        if np.any(np.linalg.eigvalsh(sigma_arr) <= 0):
            raise ValueError("sigma must be positive definite.")

        self.mu = mu_arr
        self.sigma = sigma_arr

    # ---------------------------------------------------------------------
    # Small helpers for debugging / inspection
    # ---------------------------------------------------------------------
    @property
    def dim(self) -> int:
        """
        Dimension d of latent trait vector θ.
        """
        return int(self.mu.shape[0])

    def variances(self) -> np.ndarray:
        """
        Return the marginal variances of each trait (diagonal of Sigma).
        """
        return np.diag(self.sigma)

    def stds(self) -> np.ndarray:
        """
        Return the marginal standard deviations of each trait (sqrt of variances).
        """
        return np.sqrt(self.variances())

    # ---------------------------------------------------------------------
    # Information / entropy of the Gaussian belief
    # ---------------------------------------------------------------------
    def entropy(self) -> float:
        """
        Shannon entropy of the multivariate Gaussian belief.

        Math Note:
            For N(mu, Sigma) in k dimensions,
                H = 0.5 * k * (1 + ln(2*pi)) + 0.5 * ln(det(Sigma))
              = 0.5 * ln[(2*pi*e)^k * |Sigma|]
            Lower entropy = less uncertainty = more confident estimates.
            This is the quantity we will later use as a building block for
            Information Gain when choosing which question to ask next.

        Returns:
            Scalar entropy in nats.
        """
        k = self.dim
        sign, logdet = np.linalg.slogdet(self.sigma)
        if sign <= 0:
            # This should not happen if sigma is positive definite, but we guard
            # against numerical issues.
            raise ValueError("Covariance matrix must have positive determinant.")
        # Math Note: H = 0.5 * (k * ln(2*pi*e) + ln|Sigma|). Lower entropy = tighter belief.
        return 0.5 * (k * (1 + np.log(2 * np.pi)) + logdet)

    def copy(self) -> "PersonalityState":
        """
        Return a deep copy of this state.

        Used when simulating hypothetical updates in the acquisition function
        without mutating the current belief.
        """
        return PersonalityState(mu_init=self.mu.copy(), sigma_init=self.sigma.copy())

    # ---------------------------------------------------------------------
    # Laplace-approximate update for a single Likert observation
    # ---------------------------------------------------------------------
    def update_posterior_likert_laplace(
        self,
        w: np.ndarray,
        y: int,
        thresholds: Optional[np.ndarray] = None,
        noise_var: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a Laplace-approximate posterior update for ONE Likert observation.

        Args:
            w:
                Question loading vector of shape (d,). This encodes how strongly
                the question loads on each latent trait.
            y:
                Observed Likert response as an integer in {1, ..., K}.
            thresholds:
                Array of shape (K-1,) containing strictly increasing thresholds
                τ_1 < ... < τ_{K-1}. If None, we default to a 5-point scale with
                thresholds [-1.5, -0.5, 0.5, 1.5].
            noise_var:
                Variance of the Gaussian noise in the latent score:
                    z = w^T theta + eps,  eps ~ N(0, noise_var).

        Returns:
            (new_mu, new_sigma) after the Laplace update (also stored in-place).

        Math Note (ordinal Probit model):
            - Define a 1D latent score z = w^T theta + eps, eps ~ N(0, noise_var).
            - Let thresholds τ_0 = -∞, τ_K = +∞, and τ_1 < ... < τ_{K-1}.
            - We observe category y when τ_{y-1} < z ≤ τ_y.
            - For a fixed theta, m = w^T theta is the mean of z.
              Then P(y | theta) = Φ((τ_y - m)/σ) - Φ((τ_{y-1} - m)/σ),
              where σ = sqrt(noise_var) and Φ is the Normal CDF.

        Math Note (Laplace approximation):
            - Prior: theta ~ N(mu, Sigma).
            - Posterior (up to a constant):
                log p(theta | y) = log N(theta | mu, Sigma) + log P(y | theta).
            - We find the MAP θ̂ by minimizing the negative log-posterior.
            - Around θ̂ we approximate log p(theta | y) with a quadratic:
                log p(theta | y) ≈ log p(θ̂ | y)
                  - 0.5 * (theta - θ̂)^T H (theta - θ̂),
              where H is the Hessian of the negative log-posterior at θ̂.
            - This is exactly the log-density of N(θ̂, H^{-1}), our Gaussian
              Laplace approximation to the true posterior.
        """
        # -----------------------------
        # Validate inputs
        # -----------------------------
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if w.shape[0] != self.dim:
            raise ValueError(f"w must have shape ({self.dim},), got {w.shape}.")

        if thresholds is None:
            thresholds = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
        else:
            thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if thresholds.ndim != 1 or thresholds.size == 0:
            raise ValueError("thresholds must be a non-empty 1D array.")
        if not np.all(np.diff(thresholds) > 0):
            raise ValueError("thresholds must be strictly increasing.")

        K = thresholds.size + 1
        if not (isinstance(y, int) or np.issubdtype(type(y), np.integer)):
            raise TypeError("y must be an integer Likert category.")
        if y < 1 or y > K:
            raise ValueError(f"y must be in the range [1, {K}], got {y}.")

        if noise_var <= 0:
            raise ValueError("noise_var must be positive.")

        # Precompute prior precision (inverse covariance) for efficiency.
        try:
            prior_precision = np.linalg.inv(self.sigma)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Current covariance sigma must be invertible.") from exc

        sigma_z = float(np.sqrt(noise_var))

        def _neg_log_posterior(theta: np.ndarray) -> float:
            """
            Negative log-posterior up to an additive constant.

            nlp(theta) = 0.5 * (theta - mu)^T Sigma^{-1} (theta - mu)
                         - log P(y | theta, w, thresholds)

            The constant terms of the Gaussian prior and likelihood are omitted
            since they do not affect the MAP location or the Hessian.
            """
            theta = np.asarray(theta, dtype=np.float64).reshape(-1)
            if theta.shape[0] != self.dim:
                raise ValueError("theta has incorrect dimension inside optimizer.")

            # Prior term: quadratic form from Gaussian N(mu, Sigma).
            delta = theta - self.mu
            quad = float(delta.T @ prior_precision @ delta)
            nlp_prior = 0.5 * quad

            # Likert likelihood term via ordinal Probit model.
            m = float(w @ theta)  # mean of latent z

            # Determine thresholds for this specific category y.
            if y == 1:
                a = -np.inf
                b = thresholds[0]
            elif y == K:
                a = thresholds[-1]
                b = np.inf
            else:
                a = thresholds[y - 2]
                b = thresholds[y - 1]

            alpha = (a - m) / sigma_z
            beta = (b - m) / sigma_z
            # Probability mass for this ordinal bin.
            cdf_beta = stats.norm.cdf(beta)
            cdf_alpha = stats.norm.cdf(alpha)
            p = float(cdf_beta - cdf_alpha)

            # Numerical safety: avoid log(0).
            eps = 1e-12
            if p <= 0:
                # Extremely unlikely under current parameters; assign a large penalty.
                return nlp_prior + 50.0

            nll = -np.log(p)
            return nlp_prior + nll

        # -----------------------------
        # Step 1: Find MAP theta_hat
        # -----------------------------
        theta0 = self.mu.copy()
        opt_res = optimize.minimize(
            _neg_log_posterior,
            theta0,
            method="BFGS",
        )
        if not opt_res.success:
            # We fail loudly here so issues in optimization are not silently ignored.
            raise RuntimeError(
                f"Laplace optimization failed: {opt_res.message}"
            )

        theta_hat = np.asarray(opt_res.x, dtype=np.float64).reshape(-1)

        # -----------------------------
        # Step 2: Hessian via finite differences
        # -----------------------------
        def _hessian_fd(
            f: Callable[[np.ndarray], float],
            x: np.ndarray,
            eps: float = 1e-4,
        ) -> np.ndarray:
            """
            Numerical Hessian of f at x using central finite differences.

            Math Note:
                The (i, j) entry of the Hessian H is approximated as:
                  H_ij ≈ [f(x+e_i h+e_j h) - f(x+e_i h-e_j h)
                          - f(x-e_i h+e_j h) + f(x-e_i h-e_j h)] / (4 h^2)
                For i == j this reduces to the usual 1D second derivative formula.
            """
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            n = x.size
            H = np.zeros((n, n), dtype=np.float64)

            f_x = f(x)
            # Diagonal entries
            for i in range(n):
                ei = np.zeros(n, dtype=np.float64)
                ei[i] = 1.0
                f_plus = f(x + eps * ei)
                f_minus = f(x - eps * ei)
                H[i, i] = (f_plus - 2.0 * f_x + f_minus) / (eps ** 2)

            # Off-diagonal entries
            for i in range(n):
                ei = np.zeros(n, dtype=np.float64)
                ei[i] = 1.0
                for j in range(i + 1, n):
                    ej = np.zeros(n, dtype=np.float64)
                    ej[j] = 1.0
                    f_pp = f(x + eps * ei + eps * ej)
                    f_pm = f(x + eps * ei - eps * ej)
                    f_mp = f(x - eps * ei + eps * ej)
                    f_mm = f(x - eps * ei - eps * ej)
                    val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps ** 2)
                    H[i, j] = val
                    H[j, i] = val
            return H

        H = _hessian_fd(_neg_log_posterior, theta_hat)
        # Ensure symmetry (finite-difference noise can break exact symmetry).
        H = 0.5 * (H + H.T)

        # Regularize if needed to enforce positive definiteness.
        eigvals = np.linalg.eigvalsh(H)
        min_eig = float(np.min(eigvals))
        if min_eig <= 0.0:
            # Math Note:
            #   For a valid Gaussian covariance, we need H to be positive definite.
            #   If numerical noise makes some eigenvalues non-positive, we
            #   "jitter" the Hessian by adding a small multiple of the identity
            #   to push all eigenvalues above zero.
            jitter = (1e-6 - min_eig) + 1e-6
            H = H + jitter * np.eye(self.dim, dtype=np.float64)

        try:
            new_sigma = np.linalg.inv(H)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Failed to invert Hessian for Laplace covariance.") from exc

        # Final safety check: covariance should be symmetric positive definite.
        new_sigma = 0.5 * (new_sigma + new_sigma.T)
        if np.any(np.linalg.eigvalsh(new_sigma) <= 0):
            raise RuntimeError("Laplace covariance is not positive definite.")

        # -----------------------------
        # Step 3: Commit update in-place
        # -----------------------------
        self.mu = theta_hat
        self.sigma = new_sigma

        return self.mu, self.sigma

    # ---------------------------------------------------------------------
    # Predictive distribution over Likert categories for a given question
    # ---------------------------------------------------------------------
    def predict_likert_probs(
        self,
        w: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        noise_var: float = 1.0,
    ) -> np.ndarray:
        """
        Compute predictive probabilities for Likert categories 1..K.

        Args:
            w:
                Question loading vector of shape (d,).
            thresholds:
                Array of shape (K-1,) with strictly increasing thresholds
                τ_1 < ... < τ_{K-1}. If None, defaults to a 5-point scale with
                thresholds [-1.5, -0.5, 0.5, 1.5].
            noise_var:
                Variance of the Gaussian noise in the latent score:
                    z = w^T theta + eps,  eps ~ N(0, noise_var).

        Returns:
            probs: np.ndarray of shape (K,) giving P(y = k) for k = 1..K.

        Math Note:
            Under the current belief we have theta ~ N(mu, Sigma). The question
            defines a 1D latent score
                z = w^T theta + eps.
            A linear transform of a multivariate normal is still normal, and the
            sum of independent normals is also normal, so:
                w^T theta ~ N(w^T mu, w^T Sigma w)
                eps       ~ N(0, noise_var)
                z         ~ N(mean_z, var_z)
            where:
                mean_z = w^T mu
                var_z  = w^T Sigma w + noise_var

            With thresholds τ_0 = -∞, τ_K = +∞, the probability of category k is
            the probability mass of z falling in that bin:
                P(y = k) = Φ((τ_k   - mean_z) / std_z)
                         - Φ((τ_{k-1} - mean_z) / std_z),
            where std_z = sqrt(var_z) and Φ is the standard normal CDF.
        """
        # Validate w
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if w.shape[0] != self.dim:
            raise ValueError(f"w must have shape ({self.dim},), got {w.shape}.")

        # Validate thresholds
        if thresholds is None:
            thresholds = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
        else:
            thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if thresholds.ndim != 1 or thresholds.size == 0:
            raise ValueError("thresholds must be a non-empty 1D array.")
        if not np.all(np.diff(thresholds) > 0):
            raise ValueError("thresholds must be strictly increasing.")

        if noise_var <= 0:
            raise ValueError("noise_var must be positive.")

        K = thresholds.size + 1

        # Mean and variance of z under the current Gaussian belief.
        mean_z = float(w @ self.mu)
        # Math Note:
        #   w^T theta ~ N(w^T mu, w^T Sigma w)
        #   eps ~ N(0, noise_var) is independent, so
        #   z = w^T theta + eps ~ N(mean_z, w^T Sigma w + noise_var).
        var_z = float(w @ self.sigma @ w) + float(noise_var)
        if var_z <= 0:
            raise RuntimeError("Predictive variance var_z must be positive.")
        std_z = float(np.sqrt(var_z))

        probs = np.zeros(K, dtype=np.float64)

        for k in range(1, K + 1):
            if k == 1:
                a = -np.inf
                b = thresholds[0]
            elif k == K:
                a = thresholds[-1]
                b = np.inf
            else:
                a = thresholds[k - 2]
                b = thresholds[k - 1]

            alpha = (a - mean_z) / std_z
            beta = (b - mean_z) / std_z

            cdf_beta = stats.norm.cdf(beta)
            cdf_alpha = stats.norm.cdf(alpha)
            probs[k - 1] = cdf_beta - cdf_alpha

        # Numerical safety: renormalize to ensure sum to 1.
        total = probs.sum()
        if total <= 0:
            raise RuntimeError("Predictive probabilities sum to zero; check parameters.")
        return probs / total


'''if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Very small smoke test:
    #   - create a 2D state
    #   - apply one Likert update with a question that loads mostly on
    #     the first trait
    #   - print means and variances before and after
    # This is only for quick, manual sanity checks during development.
    # ------------------------------------------------------------------
    np.set_printoptions(precision=4, suppress=True)

    state = PersonalityState(dim=2)
    print("Initial mu:", state.mu)
    print("Initial variances:", state.variances())

    # Question loading mostly on trait 0, slightly on trait 1.
    w_vec = np.array([1.0, 0.2])
    # Simulate a relatively high agreement, e.g., y = 5 on a 5-point scale.
    y_obs = 5

    new_mu, new_sigma = state.update_posterior_likert_laplace(w=w_vec, y=y_obs)

    print("Updated mu:", new_mu)
    print("Updated variances:", np.diag(new_sigma))'''

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Manual smoke tests for the Gaussian belief update.
    # These are quick sanity checks, not formal unit tests.
    # ------------------------------------------------------------------
    np.set_printoptions(precision=4, suppress=True)

    print("\n=== Smoke Test 1: Single high-agreement update ===")
    state = PersonalityState(dim=2)
    print("Initial mu:", state.mu)
    print("Initial variances:", state.variances())

    w_vec = np.array([1.0, 0.2])   # mostly trait 0
    y_obs = 5                      # strong agreement

    # Predictive probabilities before any update: neutral prior.
    probs_before = state.predict_likert_probs(w=w_vec)
    print("Predictive probs before update (neutral):", probs_before)

    new_mu, new_sigma = state.update_posterior_likert_laplace(w=w_vec, y=y_obs)

    print("Updated mu:", new_mu)
    print("Updated variances:", np.diag(new_sigma))

    # Predictive probabilities after a high-agreement update.
    probs_after = state.predict_likert_probs(w=w_vec)
    print("Predictive probs after y=5 update:", probs_after)

    print("\n=== Smoke Test 2: Repeated consistent updates ===")
    state = PersonalityState(dim=2)
    w_vec = np.array([1.0, 0.2])

    for step in range(5):
        print(f"\nStep {step + 1}")
        print("Before mu:", state.mu)
        print("Before variances:", state.variances())

        state.update_posterior_likert_laplace(w=w_vec, y=5)

        print("After mu:", state.mu)
        print("After variances:", state.variances())

    print("\n=== Smoke Test 3: Opposite-direction update ===")
    state = PersonalityState(dim=2)
    print("Initial mu:", state.mu)
    print("Initial variances:", state.variances())

    # Same question, but now low agreement
    state.update_posterior_likert_laplace(w=w_vec, y=1)

    print("Updated mu after y=1:", state.mu)
    print("Updated variances after y=1:", state.variances())

    print("\n=== Smoke Test 4: Question loading mostly on trait 1 ===")
    state = PersonalityState(dim=2)
    w_vec_trait1 = np.array([0.1, 1.0])

    print("Initial mu:", state.mu)
    print("Initial variances:", state.variances())

    state.update_posterior_likert_laplace(w=w_vec_trait1, y=5)

    print("Updated mu:", state.mu)
    print("Updated variances:", state.variances())
