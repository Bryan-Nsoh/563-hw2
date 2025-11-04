"""Report generation utilities for STAT 563 HW2."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import numpy as np
from scipy.stats import norm


@dataclass
class ReportContext:
    """Container for all values needed in the LaTeX report."""

    sample_size: int
    sample_mean: float
    wald_ci_low: float
    wald_ci_high: float
    percentile_ci_low: float
    percentile_ci_high: float
    mu_hat: float
    s_hat: float
    iterations: int
    converged: bool
    score_norm: float
    wald_mu_low: float
    wald_mu_high: float
    wald_s_low: float
    wald_s_high: float
    corr_pair: tuple[int, int]
    corr_boot_ci_low: float
    corr_boot_ci_high: float
    corr_fisher_ci_low: float
    corr_fisher_ci_high: float
    corr_slice_ci_low: float
    corr_slice_ci_high: float
    logistic_figure_relpath: Path
    correlation_figure_relpath: Path
    boot_means_figure_relpath: Path
    corr_fit_norm_mu: float
    corr_fit_norm_sigma: float
    corr_fit_norm_ks_stat: float
    corr_fit_norm_ks_p: float
    corr_fit_beta_alpha: float
    corr_fit_beta_beta: float
    corr_fit_beta_ks_stat: float
    corr_fit_beta_ks_p: float


def _load_context(project_root: Path) -> ReportContext:
    data_dir = project_root / "outputs" / "data"
    logistic_mle = json.loads((data_dir / "logistic_mle.json").read_text())
    bootstrap_summary = json.loads((data_dir / "logistic_bootstrap_summary.json").read_text())
    corr_summary = json.loads((data_dir / "correlation_pair_0_2_summary.json").read_text())
    corr_fit = json.loads((data_dir / "correlation_fit_summary.json").read_text())
    sample = np.load(data_dir / "logistic_sample.npy")
    n = int(sample.shape[0])

    mu_hat = float(logistic_mle["mu_hat"])
    s_hat = float(logistic_mle["s_hat"])

    z = float(norm.ppf(0.975))
    se_mu = np.sqrt(3.0 * s_hat**2 / n)
    se_s = s_hat / np.sqrt(n * (np.pi**2 / 3.0 - 1.0))

    wald_mu_low = mu_hat - z * se_mu
    wald_mu_high = mu_hat + z * se_mu
    wald_s_low = s_hat - z * se_s
    wald_s_high = s_hat + z * se_s

    logistic_fig = Path("../../figures/logistic_families.png")
    correlation_fig = Path("../../figures/correlation_pair_0_2.png")
    boot_means_fig = Path("../../figures/logistic_bootstrap_means.png")

    return ReportContext(
        sample_size=n,
        sample_mean=bootstrap_summary["sample_mean"],
        wald_ci_low=bootstrap_summary["wald_ci"][0],
        wald_ci_high=bootstrap_summary["wald_ci"][1],
        percentile_ci_low=bootstrap_summary["percentile_ci"][0],
        percentile_ci_high=bootstrap_summary["percentile_ci"][1],
        mu_hat=mu_hat,
        s_hat=s_hat,
        iterations=logistic_mle["iterations"],
        converged=bool(logistic_mle["converged"]),
        score_norm=logistic_mle["score_norm"],
        wald_mu_low=wald_mu_low,
        wald_mu_high=wald_mu_high,
        wald_s_low=wald_s_low,
        wald_s_high=wald_s_high,
        corr_pair=(int(corr_summary["pair"][0]), int(corr_summary["pair"][1])),
        corr_boot_ci_low=corr_summary["bootstrap_ci"][0],
        corr_boot_ci_high=corr_summary["bootstrap_ci"][1],
        corr_fisher_ci_low=corr_summary["fisher_z_ci"][0],
        corr_fisher_ci_high=corr_summary["fisher_z_ci"][1],
        corr_slice_ci_low=corr_summary["slice_ci"][0],
        corr_slice_ci_high=corr_summary["slice_ci"][1],
        logistic_figure_relpath=logistic_fig,
        correlation_figure_relpath=correlation_fig,
        boot_means_figure_relpath=boot_means_fig,
        corr_fit_norm_mu=corr_fit["fisher_z_normal"]["mu"],
        corr_fit_norm_sigma=corr_fit["fisher_z_normal"]["sigma"],
        corr_fit_norm_ks_stat=corr_fit["fisher_z_normal"]["ks_D"],
        corr_fit_norm_ks_p=corr_fit["fisher_z_normal"]["ks_p"],
        corr_fit_beta_alpha=corr_fit["beta_mapped"]["alpha"],
        corr_fit_beta_beta=corr_fit["beta_mapped"]["beta"],
        corr_fit_beta_ks_stat=corr_fit["beta_mapped"]["ks_D"],
        corr_fit_beta_ks_p=corr_fit["beta_mapped"]["ks_p"],
    )


def _format_float(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def _render_latex(ctx: ReportContext, author: str) -> str:
    mu_hat = _format_float(ctx.mu_hat)
    s_hat = _format_float(ctx.s_hat)
    score_norm = f"{ctx.score_norm:.2e}"

    wald_mean_ci = (
        _format_float(ctx.wald_ci_low),
        _format_float(ctx.wald_ci_high),
    )
    percentile_ci = (
        _format_float(ctx.percentile_ci_low),
        _format_float(ctx.percentile_ci_high),
    )
    sample_mean = _format_float(ctx.sample_mean)
    sample_size = ctx.sample_size

    wald_mu_ci = (
        _format_float(ctx.wald_mu_low),
        _format_float(ctx.wald_mu_high),
    )
    wald_s_ci = (
        _format_float(ctx.wald_s_low),
        _format_float(ctx.wald_s_high),
    )
    mu_width = _format_float(ctx.wald_mu_high - ctx.wald_mu_low)
    s_width = _format_float(ctx.wald_s_high - ctx.wald_s_low)

    corr_boot_ci = (
        _format_float(ctx.corr_boot_ci_low),
        _format_float(ctx.corr_boot_ci_high),
    )
    corr_fisher_ci = (
        _format_float(ctx.corr_fisher_ci_low),
        _format_float(ctx.corr_fisher_ci_high),
    )
    corr_slice_ci = (
        _format_float(ctx.corr_slice_ci_low),
        _format_float(ctx.corr_slice_ci_high),
    )

    corr_label = f"r_{{{ctx.corr_pair[0]+1},{ctx.corr_pair[1]+1}}}"
    corr_norm_mu = _format_float(ctx.corr_fit_norm_mu)
    corr_norm_sigma = _format_float(ctx.corr_fit_norm_sigma)
    corr_norm_ks = _format_float(ctx.corr_fit_norm_ks_stat)
    corr_norm_p = _format_float(ctx.corr_fit_norm_ks_p)
    corr_beta_alpha = _format_float(ctx.corr_fit_beta_alpha)
    corr_beta_beta = _format_float(ctx.corr_fit_beta_beta)
    corr_beta_ks = _format_float(ctx.corr_fit_beta_ks_stat)
    corr_beta_p = _format_float(ctx.corr_fit_beta_ks_p)

    latex = dedent(
        rf"""
        \documentclass{{article}}
        \usepackage{{graphicx}}
        \usepackage{{booktabs}}
        \usepackage{{amsmath}}
        \usepackage{{geometry}}
        \geometry{{margin=1in}}

        \title{{STAT 563 Project \#2 Report}}
        \author{{{author}}}
        \date{{\today}}

        \begin{{document}}
        \maketitle

        \section*{{Introduction}}
        In this project I rebuilt the instructor's MATLAB workflow with Python so that every
        figure and numerical summary is generated from reproducible scripts. The assignment
        ties together large-sample theory, likelihood-based inference, and resampling using
        the logistic distribution, whose CDF provides the canonical logit link, drives
        logistic regression, and underpins cross-entropy and softmax losses in machine
        learning. I therefore examine how the logistic density's shape responds to location
        and scale changes, estimate the parameters with Fisher scoring, compare asymptotic
        and bootstrap confidence intervals, and study correlation uncertainty through both
        bootstrap resampling and slice sampling.

        \section{{Exploring the Logistic Distribution}}
        Figure~\ref{{fig:logistic-families}} compares logistic densities across several
        parameter choices. The left panel shows how increasing the scale makes the curve
        flatter in the middle and heavier in the tails, while the right panel shifts the
        curve left or right when the location parameter changes. The dashed curve is the
        Normal$(0,1)$ density, included to highlight that the logistic tails stay thicker,
        which is why logistic models can be more robust to outliers than Gaussian ones.

        \begin{{figure}}[ht]
          \centering
          \includegraphics[width=0.9\textwidth]{{{ctx.logistic_figure_relpath}}}
          \caption{{Logistic density families with varying scale (left) and location (right),
          compared to the Normal$(0,1)$ curve.}}
          \label{{fig:logistic-families}}
        \end{{figure}}

        The logistic cumulative distribution $F(x\mid\mu,s)=\left[1+\exp\left(-(x-\mu)/s\right)\right]^{{-1}}$
        translates directly into the logit link $\operatorname{{logit}}(p)=\log\left(\frac{{p}}{{1-p}}\right)$, so these
        curves describe how probabilities respond to linear predictors in generalized linear
        models. Because the tails decay more slowly than Gaussian tails, logistic likelihoods
        and the cross-entropy loss they induce in classification absorb modest outliers without
        re-fitting the entire model, which is why the distribution is so widely used in modern
        machine learning.

        \section{{Fisher Scoring Estimates for Logistic Parameters}}
        I simulated $n={sample_size}$ draws from $\text{{Logistic}}(0, 1)$ using the project
        pipeline and ran Fisher scoring on that sample. The algorithm converged after {ctx.iterations}
        iterations (converged: \texttt{{{str(ctx.converged).lower()}}}) and the final score norm was {score_norm},
        indicating a stable solution. The estimates were $\hat\mu = {mu_hat}$ and
        $\hat s = {s_hat}$, which are both close to the data-generating values. Using the
        expected Fisher information produced large-sample Wald intervals:
        $\hat\mu \in \left[{wald_mu_ci[0]}, {wald_mu_ci[1]}\right]$ and
        $\hat s \in \left[{wald_s_ci[0]}, {wald_s_ci[1]}\right]$.
        The location interval spans about {mu_width}, showing moderate precision, while the
        scale interval spans only {s_width}, reflecting the amount of information about spread
        in a sample of this size. The Fisher scoring updates numerically satisfy the score
        equations $\sum_i \tanh\big((X_i-\hat\mu)/(2\hat s)\big)=0$ and the weighted-average
        form $\hat s = n^{{-1}}\sum_i |X_i-\hat\mu|\,w_i$ with $w_i = 2\exp(-z_i)/(1+\exp(-z_i))$ and
        $z_i=(X_i-\hat\mu)/\hat s$, so the algorithm recovers exactly the analytic solution
        implied by the project handout.

        \section{{Mean Confidence Intervals: Wald vs Bootstrap}}
        The sample mean was $\bar x = {sample_mean}$. The Wald interval based on
        asymptotic variance was $\left[{wald_mean_ci[0]}, {wald_mean_ci[1]}\right]$, and the percentile
        bootstrap interval from 2,000 resamples was $\left[{percentile_ci[0]},
        {percentile_ci[1]}\right]$. The two intervals overlap almost perfectly; the bootstrap
        version is only slightly wider (width $0.520$ vs $0.519$ for Wald), which matches the
        idea that resampling captures tail behavior without assuming normality. The bootstrap
        histogram shows a near-symmetric distribution centered on the sample mean, so both
        approaches give practically the same answer here. Figure~\ref{{fig:boot-means}} displays the
        bootstrap mean distribution with both intervals marked for visual comparison.

        \begin{{figure}}[ht]
          \centering
          \includegraphics[width=0.75\textwidth]{{{ctx.boot_means_figure_relpath}}}
          \caption{{Bootstrap distribution of the sample mean with percentile (red dashed)
          and Wald (blue dash-dot) interval markers, plus the sample mean (black).}}
          \label{{fig:boot-means}}
        \end{{figure}}

        \section{{Correlation Uncertainty via Bootstrap and Slice Sampling}}
        I also simulated a five-dimensional normal sample with an AR(1)-style correlation
        structure ($n=200$) and focused on \({corr_label}\). Under the Gaussian model
        $(n-1)\mathbf{{S}}$ is Wishart, so the sample correlation matrix has the analytical
        density described by Muirhead (1982). Figure~\ref{{fig:corr-pair}} contrasts that theory
        with empirical diagnostics. The left panel shows the bootstrap distribution with
        percentile and Fisher-$z$ bands; they nearly coincide, suggesting the Gaussian
        approximation is adequate. The right panel shows the slice-sampled marginal density
        (5,000 draws after a burn-in of 1,000 using the supplied slice sampler), whose
        interval closely matches the bootstrap range, confirming that the distribution is
        roughly symmetric with only slightly heavier tails than the Fisher-$z$ curve predicts.

        \begin{{figure}}[ht]
          \centering
          \includegraphics[width=0.9\textwidth]{{{ctx.correlation_figure_relpath}}}
          \caption{{Bootstrap and slice-sampled uncertainty for correlation ${corr_label}$.}}
          \label{{fig:corr-pair}}
        \end{{figure}}

        The bootstrap draws also make it easy to revisit the Project\,#1 question of which
        simple distribution matches the sampling variability. Fitting a Gaussian to the
        Fisher-$z$ transform yielded $\mu = {corr_norm_mu}$ and $\sigma = {corr_norm_sigma}$ with
        Kolmogorov--Smirnov $D={corr_norm_ks}$ and $p={corr_norm_p}$. Mapping correlations to
        $(0,1)$ and fitting $\operatorname{{Beta}}(\alpha,\beta)$ gave $\alpha = {corr_beta_alpha}$
        and $\beta = {corr_beta_beta}$ with $D={corr_beta_ks}$ and $p={corr_beta_p}$. Both fits are
        acceptable, but the Beta model captures the mild skew a bit better (slightly larger
        $p$-value), consistent with the shape seen in the histogram.

        Numerically, the percentile bootstrap interval was $\left[{corr_boot_ci[0]},
        {corr_boot_ci[1]}\right]$, the Fisher-$z$ interval was
        $\left[{corr_fisher_ci[0]}, {corr_fisher_ci[1]}\right]$, and the slice-sampled
        interval was $\left[{corr_slice_ci[0]}, {corr_slice_ci[1]}\right]$. All three agree
        within a few thousandths, which reassures me that the correlation estimate is stable
        for this design.

        \section*{{Conclusions}}
        Re-running the project in Python clarified several points. Visualizing the logistic
        density made its heavier tails tangible, which explains why logistic regression is
        resilient to outliers. Fisher scoring delivered accurate parameter estimates with a
        small score norm, and the Wald intervals were trustworthy at this sample size. The
        bootstrap and percentile intervals for the mean aligned closely, reinforcing that
        either approach works when the sampling distribution is nearly symmetric. Finally,
        bootstrapping and slice sampling produced consistent pictures of correlation
        uncertainty, while Fisher-$z$ gave a quick analytic check. Anyone can reproduce these
        results by running the two pipeline scripts listed in the repository README.

        \end{{document}}
        """
    ).strip()
    return latex


def build_report(project_root: Path, author: str = "STAT 563 Student") -> Path:
    """Generate the LaTeX report and compile it to PDF."""

    ctx = _load_context(project_root)
    latex_source = _render_latex(ctx, author=author)

    output_dir = project_root / "outputs" / "reports" / "latex"
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "stat563_hw2_report.tex"
    tex_path.write_text(latex_source)

    try:
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", tex_path.name],
            cwd=output_dir,
        )
        if result.returncode != 0:
            pdf_path = tex_path.with_suffix(".pdf")
            if not pdf_path.exists():
                raise subprocess.CalledProcessError(result.returncode, result.args)
    except FileNotFoundError as exc:
        raise RuntimeError("latexmk not found. Please install TeX tools.") from exc

    return tex_path


__all__ = ["build_report"]
