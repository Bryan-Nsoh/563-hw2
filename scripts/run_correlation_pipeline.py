"""Execute the correlation-matrix bootstrap + slice sampling workflow (Q4)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz

from stat563_hw2.correlation import (
    analyse_correlation_pair,
    bootstrap_correlation_vectors,
)

OUTPUT_DATA = PROJECT_ROOT / "outputs" / "data"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _figure_for_pair(
    r_boot: np.ndarray,
    r_slice: np.ndarray,
    *,
    bootstrap_ci: Tuple[float, float],
    fisher_ci: Tuple[float, float],
    slice_ci: Tuple[float, float],
    pair: Tuple[int, int],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    ax = axes[0]
    ax.hist(r_boot, bins=40, density=True, color="#4477AA", alpha=0.7, edgecolor="none")
    ax.axvline(bootstrap_ci[0], color="red", linestyle="--", label="Boot 2.5%")
    ax.axvline(bootstrap_ci[1], color="red", linestyle="--", label="Boot 97.5%")
    ax.axvline(fisher_ci[0], color="black", linestyle="-.", label="Fisher 2.5%")
    ax.axvline(fisher_ci[1], color="black", linestyle="-.", label="Fisher 97.5%")
    ax.set_title(f"Bootstrap distribution r_{{{pair[0]+1},{pair[1]+1}}}")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.hist(r_slice, bins=40, density=True, color="#66CCAA", alpha=0.7, edgecolor="none")
    ax.axvline(slice_ci[0], color="magenta", linestyle="--", label="Slice 2.5%")
    ax.axvline(slice_ci[1], color="magenta", linestyle="--", label="Slice 97.5%")
    ax.set_title(f"Slice-sampled marginal r_{{{pair[0]+1},{pair[1]+1}}}")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run() -> None:
    rng = np.random.default_rng(563)

    n = 200
    d = 5
    sigma = toeplitz(0.7 ** np.arange(d))
    data = rng.multivariate_normal(mean=np.zeros(d), cov=sigma, size=n)

    bootstrap_matrix, pairs = bootstrap_correlation_vectors(data, rng=rng)
    np.save(OUTPUT_DATA / "correlation_bootstrap_vectors.npy", bootstrap_matrix)
    np.save(OUTPUT_DATA / "correlation_pairs.npy", pairs)
    np.save(OUTPUT_DATA / "correlation_sample.npy", data)

    target_pair = (0, 2)  # corresponds to MATLAB's j=1, k=3 (1-based)
    pair_index = int(np.where((pairs == target_pair).all(axis=1))[0][0])

    result = analyse_correlation_pair(
        bootstrap_matrix,
        pairs,
        pair_index,
        sample_size=n,
        rng=rng,
    )

    summary = {
        "pair": [int(v) for v in result.pair],
        "bootstrap_ci": list(result.bootstrap_ci),
        "fisher_z_ci": list(result.fisher_z_ci),
        "slice_ci": list(result.slice_ci),
        "bootstrap_samples_path": str((OUTPUT_DATA / "correlation_bootstrap_vectors.npy").relative_to(PROJECT_ROOT)),
        "slice_samples_path": str((OUTPUT_DATA / "correlation_slice_samples_pair_0_2.npy").relative_to(PROJECT_ROOT)),
    }
    save_json(OUTPUT_DATA / "correlation_pair_0_2_summary.json", summary)

    np.save(OUTPUT_DATA / "correlation_slice_samples_pair_0_2.npy", result.slice_samples)

    figure_path = OUTPUT_FIGURES / "correlation_pair_0_2.png"
    _figure_for_pair(
        result.bootstrap_samples,
        result.slice_samples,
        bootstrap_ci=result.bootstrap_ci,
        fisher_ci=result.fisher_z_ci,
        slice_ci=result.slice_ci,
        pair=result.pair,
        output_path=figure_path,
    )

    print("Correlation pipeline completed:")
    print("  bootstrap matrix -> outputs/data/correlation_bootstrap_vectors.npy")
    print("  pair summary -> outputs/data/correlation_pair_0_2_summary.json")
    print(f"  figure -> {figure_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run()
