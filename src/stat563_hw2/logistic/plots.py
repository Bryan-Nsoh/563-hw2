"""Plotting utilities for the logistic distribution."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def logistic_pdf(x: np.ndarray, mu: float, s: float) -> np.ndarray:
    """Return the logistic probability density function.

    Parameters
    ----------
    x : np.ndarray
        Evaluation points.
    mu : float
        Location parameter.
    s : float
        Scale parameter (positive).
    """

    z = (x - mu) / s
    return np.exp(-z) / (s * (1.0 + np.exp(-z)) ** 2)


def plot_logistic_families(
    output_path: Path,
    x_bounds: tuple[float, float] = (-8.0, 8.0),
    num_points: int = 400,
    scale_values: Sequence[float] = (0.5, 1.0, 2.0),
    location_values: Sequence[float] = (-2.0, 0.0, 2.0),
    figsize: tuple[int, int] = (10, 6),
    dpi: int = 150,
) -> Path:
    """Create the diagnostic logistic plots required in Q1.

    The figure contains one panel showing scale variation and one panel showing
    location shifts. Each panel also overlays the standard normal density for
    comparison.

    Parameters
    ----------
    output_path : Path
        File path to save the figure (PNG recommended).
    x_bounds : tuple[float, float]
        Lower and upper x-limits for evaluation.
    num_points : int
        Number of points in the grid.
    scale_values : Sequence[float]
        Scale parameters to showcase when mu=0.
    location_values : Sequence[float]
        Location parameters to showcase when s=1.
    figsize : tuple[int, int]
        Matplotlib figure size in inches.
    dpi : int
        Figure resolution.

    Returns
    -------
    Path
        The output path, for convenience.
    """

    x = np.linspace(*x_bounds, num_points)
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(scale_values), len(location_values))))

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharey=True)

    ax = axes[0]
    for color, s in zip(colors, scale_values):
        ax.plot(x, logistic_pdf(x, 0.0, s), color=color, lw=2, label=rf"$\mu=0,\ s={s}$")
    ax.plot(x, norm.pdf(x, loc=0.0, scale=1.0), color="black", ls="--", lw=1.5, label="Normal(0,1)")
    ax.set_title("Scale effects")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for color, mu in zip(colors, location_values):
        ax.plot(x, logistic_pdf(x, mu, 1.0), color=color, lw=2, label=rf"$\mu={mu},\ s=1$")
    ax.plot(x, norm.pdf(x, loc=0.0, scale=1.0), color="black", ls="--", lw=1.5, label="Normal(0,1)")
    ax.set_title("Location effects")
    ax.set_xlabel("x")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Logistic density families vs Normal(0,1)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


__all__ = ["plot_logistic_families", "logistic_pdf"]
