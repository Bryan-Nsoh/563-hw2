"""Bootstrap procedures for the logistic distribution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


@dataclass
class BootstrapResult:
    """Summary of bootstrap vs Wald confidence intervals."""

    sample_mean: float
    wald_ci: tuple[float, float]
    percentile_ci: tuple[float, float]
    bootstrap_means: NDArray[np.float64]


def _logistic_scale_from_sample(sample: NDArray[np.float64]) -> float:
    """Infer the logistic scale using the plug-in variance estimate."""

    variance = np.var(sample, ddof=1)
    if variance < 0:
        raise RuntimeError("Sample variance should not be negative.")
    return np.sqrt(variance * 3.0 / np.pi**2)


def wald_mean_ci(
    sample: NDArray[np.float64],
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute the Wald CI for the logistic mean using Fisher information."""

    n = len(sample)
    if n == 0:
        raise ValueError("Sample cannot be empty.")

    s_hat = _logistic_scale_from_sample(sample)
    z = norm.ppf(1.0 - alpha / 2.0)
    half_width = z * np.sqrt((np.pi**2 * s_hat**2) / (3.0 * n))
    mean = float(np.mean(sample))
    return mean - half_width, mean + half_width


def bootstrap_mean_ci(
    sample: NDArray[np.float64],
    *,
    num_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> BootstrapResult:
    """Bootstrap percentile CI for the mean of the logistic distribution."""

    x = np.asarray(sample, dtype=float)
    if x.ndim != 1:
        raise ValueError("Sample must be one-dimensional.")
    n = len(x)
    if n == 0:
        raise ValueError("Sample cannot be empty.")
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be positive.")

    rng = np.random.default_rng() if rng is None else rng
    means = np.empty(num_bootstrap, dtype=float)
    for b in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[b] = float(np.mean(x[idx]))

    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0
    percentile_ci = (float(np.quantile(means, lower_q)), float(np.quantile(means, upper_q)))
    wald_ci = wald_mean_ci(x, alpha=alpha)
    return BootstrapResult(
        sample_mean=float(np.mean(x)),
        wald_ci=wald_ci,
        percentile_ci=percentile_ci,
        bootstrap_means=means,
    )


__all__ = [
    "BootstrapResult",
    "bootstrap_mean_ci",
    "wald_mean_ci",
]
