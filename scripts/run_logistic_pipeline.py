"""Execute the logistic-distribution experiments (Q1-Q3)."""
from __future__ import annotations

import json
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

    print("Logistic pipeline completed:")
    print(f"  figure -> {figure_path.relative_to(PROJECT_ROOT)}")
    print("  mle -> outputs/data/logistic_mle.json")
    print("  bootstrap summary -> outputs/data/logistic_bootstrap_summary.json")


if __name__ == "__main__":
    run()
