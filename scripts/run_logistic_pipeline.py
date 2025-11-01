"""Execute the logistic-distribution experiments (Q1-Q3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np

from stat563_hw2.logistic import (
    BootstrapResult,
    FisherScoringResult,
    bootstrap_mean_ci,
    fisher_scoring_logistic_mle,
    plot_logistic_families,
    simulate_logistic_sample,
    wald_mean_ci,
)

OUTPUT_DATA = PROJECT_ROOT / "outputs" / "data"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def run() -> None:
    rng = np.random.default_rng(563)

    figure_path = OUTPUT_FIGURES / "logistic_families.png"
    plot_logistic_families(figure_path)

    sample = simulate_logistic_sample(200, mu=0.0, s=1.0, rng=rng)
    mle_result: FisherScoringResult = fisher_scoring_logistic_mle(sample)
    wald_ci = wald_mean_ci(sample)
    bootstrap_result: BootstrapResult = bootstrap_mean_ci(sample, rng=rng)

    mle_payload = {
        "mu_hat": mle_result.mu,
        "s_hat": mle_result.s,
        "iterations": mle_result.iterations,
        "converged": mle_result.converged,
        "score_norm": mle_result.score_norm,
    }
    save_json(OUTPUT_DATA / "logistic_mle.json", mle_payload)

    bootstrap_payload = {
        "sample_mean": bootstrap_result.sample_mean,
        "wald_ci": list(wald_ci),
        "wald_ci_internal": list(bootstrap_result.wald_ci),
        "percentile_ci": list(bootstrap_result.percentile_ci),
        "bootstrap_means_path": str((OUTPUT_DATA / "logistic_bootstrap_means.npy").relative_to(PROJECT_ROOT)),
    }
    save_json(OUTPUT_DATA / "logistic_bootstrap_summary.json", bootstrap_payload)

    np.save(OUTPUT_DATA / "logistic_sample.npy", sample)
    np.save(OUTPUT_DATA / "logistic_bootstrap_means.npy", bootstrap_result.bootstrap_means)

    # Figure: bootstrap means histogram with CI markers
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(bootstrap_result.bootstrap_means, bins=40, density=True, color="#6aaed6", edgecolor="none")
    ax.axvline(bootstrap_result.sample_mean, color="k", lw=1.5, label="sample mean")
    ax.axvline(bootstrap_result.percentile_ci[0], color="r", ls="--", lw=1.2, label="bootstrap 2.5%")
    ax.axvline(bootstrap_result.percentile_ci[1], color="r", ls="--", lw=1.2, label="bootstrap 97.5%")
    ax.axvline(wald_ci[0], color="b", ls="-.", lw=1.2, label="Wald 2.5%")
    ax.axvline(wald_ci[1], color="b", ls="-.", lw=1.2, label="Wald 97.5%")
    ax.set_xlabel("Bootstrap means")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    boot_fig = OUTPUT_FIGURES / "logistic_bootstrap_means.png"
    fig.savefig(boot_fig)
    plt.close(fig)

    print("Logistic pipeline completed:")
    print(f"  figure -> {figure_path.relative_to(PROJECT_ROOT)}")
    print("  mle -> outputs/data/logistic_mle.json")
    print("  bootstrap summary -> outputs/data/logistic_bootstrap_summary.json")
    print("  bootstrap means fig -> outputs/figures/logistic_bootstrap_means.png")


if __name__ == "__main__":
    run()
