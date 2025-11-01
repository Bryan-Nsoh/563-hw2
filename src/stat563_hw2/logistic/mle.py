"""Maximum likelihood estimation for the logistic distribution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit


@dataclass
class FisherScoringResult:
    """Container for the Fisher scoring output."""

    mu: float
    s: float
    iterations: int
    converged: bool
    score_norm: float
    history: List[Tuple[float, float]] = field(default_factory=list)


def simulate_logistic_sample(
    n: int,
    mu: float = 0.0,
    s: float = 1.0,
    *,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """Draw iid samples from Logistic(mu, s) using inverse transform sampling."""

    if n <= 0:
        raise ValueError("Sample size n must be positive.")
    if s <= 0:
        raise ValueError("Scale parameter s must be positive.")

    rng = np.random.default_rng() if rng is None else rng
    u = rng.random(n)
    u = np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return mu + s * np.log(u / (1 - u))


def _score(sample: NDArray[np.float64], mu: float, s: float) -> NDArray[np.float64]:
    """Compute the score vector (gradient of log-likelihood)."""

    if s <= 0:
        raise ValueError("Scale parameter must remain positive during optimisation.")

    z = (sample - mu) / s
    tanh_term = np.tanh(z / 2.0)
    logistic_neg = expit(-z)  # equals exp(-z) / (1 + exp(-z))

    score_mu = np.sum(tanh_term) / s
    score_s = (
        -len(sample) / s
        + np.sum(sample - mu) / (s**2)
        - (2.0 / s**2) * np.sum((sample - mu) * logistic_neg)
    )
    return np.array([score_mu, score_s])


def _expected_information(n: int, s: float) -> NDArray[np.float64]:
    """Return the expected Fisher information matrix at (mu, s)."""

    if s <= 0:
        raise ValueError("Scale parameter must be positive.")

    info_mu = n / (3.0 * s**2)
    info_s = n * (np.pi**2 / 3.0 - 1.0) / (s**2)
    return np.array([[info_mu, 0.0], [0.0, info_s]])


def fisher_scoring_logistic_mle(
    sample: ArrayLike,
    *,
    mu_init: Optional[float] = None,
    s_init: Optional[float] = None,
    tol: float = 1e-8,
    max_iter: int = 200,
    min_scale: float = 1e-6,
) -> FisherScoringResult:
    """Estimate (mu, s) via Fisher scoring using the expected information matrix."""

    x = np.asarray(sample, dtype=float)
    if x.ndim != 1:
        raise ValueError("Sample must be one-dimensional.")
    if len(x) == 0:
        raise ValueError("Sample cannot be empty.")

    mu = np.median(x) if mu_init is None else float(mu_init)
    if s_init is None:
        mad = np.median(np.abs(x - mu))
        s = max(mad / np.log(3), min_scale)  # logistic MAD factor
    else:
        s = max(float(s_init), min_scale)

    history: List[Tuple[float, float]] = [(mu, s)]

    for iteration in range(1, max_iter + 1):
        score = _score(x, mu, s)
        info = _expected_information(len(x), s)
        try:
            delta = np.linalg.solve(info, score)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Fisher information matrix is singular.") from exc

        mu_next = mu + delta[0]
        s_next = s + delta[1]

        step = 0
        while s_next <= min_scale and step < 20:
            delta *= 0.5
            mu_next = mu + delta[0]
            s_next = s + delta[1]
            step += 1

        if s_next <= min_scale:
            s_next = min_scale

        mu, s = mu_next, s_next
        history.append((mu, s))

        if np.linalg.norm(delta) < tol:
            return FisherScoringResult(
                mu=mu,
                s=s,
                iterations=iteration,
                converged=True,
                score_norm=float(np.linalg.norm(score)),
                history=history,
            )

    return FisherScoringResult(
        mu=mu,
        s=s,
        iterations=max_iter,
        converged=False,
        score_norm=float(np.linalg.norm(_score(x, mu, s))),
        history=history,
    )


__all__ = [
    "FisherScoringResult",
    "fisher_scoring_logistic_mle",
    "simulate_logistic_sample",
]
