"""Bootstrap utilities for correlation matrices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import gaussian_kde, norm

from .slice_sampling import slice_sample_1d


@dataclass
class BootstrapCorrelationResult:
    """Summary statistics for a single correlation pair."""

    pair: Tuple[int, int]
    bootstrap_ci: Tuple[float, float]
    fisher_z_ci: Tuple[float, float]
    slice_ci: Tuple[float, float]
    bootstrap_samples: NDArray[np.float64]
    slice_samples: NDArray[np.float64]


def bootstrap_correlation_vectors(
    data: ArrayLike,
    *,
    num_bootstrap: int = 1500,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], np.ndarray]:
    """Resample the correlation matrix and return vectorised upper triangles."""

    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("Input data must be a 2D array (n, d).")

    n, d = x.shape
    if n < 2 or d < 2:
        raise ValueError("Need at least 2 observations and 2 variables.")
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be positive.")

    rng = np.random.default_rng() if rng is None else rng
    upper_idx = np.triu_indices(d, k=1)
    num_pairs = upper_idx[0].size

    samples = np.empty((num_bootstrap, num_pairs), dtype=float)
    for b in range(num_bootstrap):
        resample_idx = rng.integers(0, n, size=n)
        xb = x[resample_idx, :]
        corr = np.corrcoef(xb, rowvar=False)
        samples[b, :] = corr[upper_idx]

    pairs = np.column_stack(upper_idx).astype(np.int_)
    return samples, pairs


def bootstrap_percentile_ci(
    samples: NDArray[np.float64],
    *,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Simple percentile interval from bootstrap draws."""

    lower = float(np.quantile(samples, alpha / 2.0))
    upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lower, upper


def fisher_z_interval(
    r_draws: NDArray[np.float64],
    n: int,
    *,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Fisher-z interval centred at the bootstrap average (matches MATLAB script)."""

    if n <= 3:
        raise ValueError("Sample size must exceed 3 for Fisher-z intervals.")

    z = np.arctanh(np.clip(r_draws, -0.999999, 0.999999))
    z_mean = float(np.mean(z))
    z_se = 1.0 / np.sqrt(n - 3.0)
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    z_ci = (z_mean - z_crit * z_se, z_mean + z_crit * z_se)
    return tuple(np.tanh(z_ci))  # type: ignore[return-value]


def analyse_correlation_pair(
    bootstrap_matrix: NDArray[np.float64],
    pairs: NDArray[np.int_],
    pair_index: int,
    *,
    sample_size: int,
    alpha: float = 0.05,
    slice_samples: int = 5000,
    burn_in: int = 1000,
    slice_width: float = 0.05,
    slice_max_steps: int = 50,
    bounds: Tuple[float, float] = (-0.999, 0.999),
    rng: Optional[np.random.Generator] = None,
) -> BootstrapCorrelationResult:
    """Replicate the MATLAB bootstrap + slice sampling workflow for one pair."""

    if pair_index < 0 or pair_index >= bootstrap_matrix.shape[1]:
        raise IndexError("pair_index is out of range for the bootstrap matrix.")

    rng = np.random.default_rng() if rng is None else rng
    r_boot = bootstrap_matrix[:, pair_index]

    kde = gaussian_kde(r_boot)

    def logpdf(value: float) -> float:
        if value <= bounds[0] or value >= bounds[1]:
            return -np.inf
        density = float(kde.evaluate([value]))
        if density <= 0:
            return -np.inf
        return np.log(density)

    initial = float(np.median(r_boot))
    slice_draws = slice_sample_1d(
        logpdf,
        initial,
        width=slice_width,
        max_steps=slice_max_steps,
        bounds=bounds,
        num_samples=slice_samples,
        burn_in=burn_in,
        rng=rng,
    )

    boot_ci = bootstrap_percentile_ci(r_boot, alpha=alpha)
    fisher_ci = fisher_z_interval(r_boot, sample_size, alpha=alpha)
    slice_ci = (
        float(np.quantile(slice_draws, alpha / 2.0)),
        float(np.quantile(slice_draws, 1.0 - alpha / 2.0)),
    )

    pair_tuple = tuple(int(v) for v in pairs[pair_index, :])  # type: ignore[assignment]

    return BootstrapCorrelationResult(
        pair=pair_tuple,  # type: ignore[arg-type]
        bootstrap_ci=boot_ci,
        fisher_z_ci=fisher_ci,
        slice_ci=slice_ci,
        bootstrap_samples=r_boot,
        slice_samples=slice_draws,
    )


__all__ = [
    "BootstrapCorrelationResult",
    "analyse_correlation_pair",
    "bootstrap_correlation_vectors",
    "bootstrap_percentile_ci",
    "fisher_z_interval",
]
