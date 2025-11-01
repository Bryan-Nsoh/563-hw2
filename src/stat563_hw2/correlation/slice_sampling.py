"""Univariate slice sampling utilities."""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


def slice_sample_1d(
    logpdf: Callable[[float], float],
    initial: float,
    *,
    width: float = 0.05,
    max_steps: int = 50,
    bounds: Tuple[float, float] = (-0.999, 0.999),
    num_samples: int = 5000,
    burn_in: int = 1000,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Perform univariate slice sampling with stepping out and shrinkage."""

    if width <= 0:
        raise ValueError("width must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")

    lower, upper = bounds
    if lower >= upper:
        raise ValueError("bounds must satisfy lower < upper")

    rng = np.random.default_rng() if rng is None else rng

    x = float(np.clip(initial, lower, upper))
    draws = np.empty(num_samples + burn_in, dtype=float)

    for t in range(num_samples + burn_in):
        logy = logpdf(x) + np.log(rng.random())
        u = rng.random() * width
        a = x - u
        b = x + (width - u)

        j = int(rng.integers(0, max_steps))
        k = max_steps - 1 - j

        while j > 0 and a > lower:
            if logpdf(a) <= logy:
                break
            a -= width
            j -= 1
        a = max(a, lower)

        while k > 0 and b < upper:
            if logpdf(b) <= logy:
                break
            b += width
            k -= 1
        b = min(b, upper)

        while True:
            x_new = rng.uniform(a, b)
            if logpdf(x_new) >= logy:
                x = x_new
                break
            if x_new < x:
                a = x_new
            else:
                b = x_new

        draws[t] = x

    return draws[burn_in:]


__all__ = ["slice_sample_1d"]
