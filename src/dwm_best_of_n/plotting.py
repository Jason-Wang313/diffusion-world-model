"""Paper-quality plotting helpers with stable artifact names."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "imagined": "#1f77b4",
    "real": "#d62728",
    "raw": "#d62728",
    "calibrated": "#2ca02c",
    "uncertainty_aware": "#9467bd",
    "consistency_aware": "#8c564b",
    "random": "#7f7f7f",
    "oracle": "#111111",
    "pilot_lcb": "#2ca02c",
    "repair_oracle_features": "#111111",
    "repair_many_pilot_labels": "#ff7f0e",
}


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def figure1_tail_hallucination(metrics: pd.DataFrame, figures_dir: str | Path) -> Path:
    df = metrics[
        (metrics["experiment"] == "controlled")
        & (metrics["generator"] == "optimistic")
        & (metrics["scorer"] == "raw")
    ].sort_values("N")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(df["N"], df["selected_imagined_score_mean"], yerr=df["selected_imagined_score_ci95"], marker="o", color=COLORS["imagined"], label="selected imagined score")
    ax.errorbar(df["N"], df["selected_real_utility_mean"], yerr=df["selected_real_utility_ci95"], marker="s", color=COLORS["real"], label="selected real utility")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N generated futures")
    ax.set_ylabel("Mean selected value")
    ax.set_title("Tail hallucination under raw diffusion-world scoring")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = Path(figures_dir) / "figure1_tail_hallucination.png"
    _finish(fig, path)
    return path


def figure2_repair_comparison(metrics: pd.DataFrame, figures_dir: str | Path) -> Path:
    df = metrics[(metrics["experiment"] == "repair") & (metrics["generator"] == "optimistic")]
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for scorer in ["raw", "calibrated", "uncertainty_aware", "consistency_aware", "random", "oracle"]:
        part = df[df["scorer"] == scorer].sort_values("N")
        if part.empty:
            continue
        ax.plot(part["N"], part["selected_real_utility_mean"], marker="o", label=scorer, color=COLORS.get(scorer))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N generated futures")
    ax.set_ylabel("Selected real utility")
    ax.set_title("Controlled selected-tail repair")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    path = Path(figures_dir) / "figure2_repair_comparison.png"
    _finish(fig, path)
    return path


def figure3_tail_diagnostics(metrics: pd.DataFrame, figures_dir: str | Path) -> Path:
    df = metrics[(metrics["experiment"] == "controlled") & (metrics["scorer"] == "raw")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharex=True)
    for gen in df["generator"].unique():
        part = df[df["generator"] == gen].sort_values("N")
        axes[0].plot(part["N"], part["imagined_real_tail_gap"], marker="o", label=gen)
        axes[1].plot(part["N"], part["upper_tail_rank_correlation"], marker="o", label=gen)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("N generated futures")
    axes[0].set_ylabel("Imagined-real tail gap")
    axes[1].set_ylabel("Upper-tail rank correlation")
    axes[0].set_title("Tail gap")
    axes[1].set_title("Tail rank distortion")
    axes[1].legend(frameon=False, fontsize=8)
    path = Path(figures_dir) / "figure3_tail_diagnostics.png"
    _finish(fig, path)
    return path


def figure4_denoising_vs_selection(grid: pd.DataFrame, figures_dir: str | Path) -> Path:
    pivot = grid.pivot_table(index="denoising_steps", columns="N", values="selected_real_utility_mean", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), labels=[str(i) for i in pivot.index])
    ax.set_xlabel("N generated futures")
    ax.set_ylabel("Denoising steps K")
    ax.set_title("Denoising budget vs selection budget")
    for i, k in enumerate(pivot.index):
        for j, n in enumerate(pivot.columns):
            ax.text(j, i, f"{pivot.loc[k, n]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="Selected real utility")
    path = Path(figures_dir) / "figure4_denoising_vs_selection.png"
    _finish(fig, path)
    return path


def figure5_exact_law_validation(validation: pd.DataFrame, figures_dir: str | Path) -> Path:
    df = validation.sort_values("N")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(df["N"], df["law_expected_utility"], marker="o", label="exact finite law", color="#1f77b4")
    ax.plot(df["N"], df["mc_expected_utility"], marker="s", label="Monte Carlo", color="#ff7f0e")
    ax.fill_between(
        df["N"],
        df["mc_expected_utility"] - 2 * df["mc_se_utility"],
        df["mc_expected_utility"] + 2 * df["mc_se_utility"],
        color="#ff7f0e",
        alpha=0.18,
        linewidth=0,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N generated futures")
    ax.set_ylabel("Expected selected utility")
    ax.set_title("Tie-aware finite Best-of-N law validation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = Path(figures_dir) / "figure5_exact_law_validation.png"
    _finish(fig, path)
    return path


def figure6_pilot_repair_gap_closure(gap: pd.DataFrame, figures_dir: str | Path) -> Path:
    df = gap[
        (gap["repair_model"] == "pilot_lcb")
        & (gap["experiment"].isin(["controlled_pilot_repair", "learned_pilot_repair"]))
    ].sort_values(["experiment", "pilot_budget"])
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for experiment, label, color in [
        ("controlled_pilot_repair", "controlled optimistic", "#2ca02c"),
        ("learned_pilot_repair", "learned held-out", "#1f77b4"),
    ]:
        part = df[df["experiment"] == experiment]
        if part.empty:
            continue
        ax.plot(part["pilot_budget"], part["gap_closed_clamped"], marker="o", label=label, color=color)
    ax.axhline(0.70, color="#555555", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xscale("symlog", linthresh=8)
    ax.set_xlabel("Pilot real-rollout labels")
    ax.set_ylabel("Oracle gap closed")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Pilot-label LCB repair closes selected-tail gap")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = Path(figures_dir) / "figure6_pilot_repair_gap_closure.png"
    _finish(fig, path)
    return path


def figure7_adaptive_n_gate(adaptive: pd.DataFrame, figures_dir: str | Path) -> Path:
    df = adaptive[
        (adaptive["experiment"] == "controlled_pilot_repair")
        & (adaptive["repair_model"] == "pilot_lcb")
        & (adaptive["pilot_budget"].isin([0, 32, 128]))
    ].sort_values(["pilot_budget", "N"])
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    markers = {
        "allow_high_n": "o",
        "stop_early": "s",
        "collect_pilot_labels": "^",
        "block_high_n": "x",
    }
    colors = {0: "#7f7f7f", 32: "#2ca02c", 128: "#111111"}
    for budget, part in df.groupby("pilot_budget"):
        ax.plot(part["N"], part["selected_lcb_mean"], color=colors.get(int(budget), "#1f77b4"), linewidth=1.5, label=f"budget {int(budget)}")
        for decision, marker in markers.items():
            points = part[part["deployment_gate"] == decision]
            if not points.empty:
                ax.scatter(points["N"], points["selected_lcb_mean"], marker=marker, color=colors.get(int(budget), "#1f77b4"), s=42)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N generated futures")
    ax.set_ylabel("Selected lower-confidence estimate")
    ax.set_title("Adaptive Best-of-N gate")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    path = Path(figures_dir) / "figure7_adaptive_n_gate.png"
    _finish(fig, path)
    return path


def figure8_calibration_reliability(calibration: pd.DataFrame, figures_dir: str | Path) -> Path:
    df = calibration[calibration["repair_model"] == "pilot_lcb"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9))
    for experiment, label, color in [
        ("controlled_pilot_repair", "controlled", "#2ca02c"),
        ("learned_pilot_repair", "learned", "#1f77b4"),
    ]:
        part = df[df["experiment"] == experiment].sort_values("pilot_budget")
        if part.empty:
            continue
        axes[0].plot(part["pilot_budget"], part["eval_lower_bound_coverage"], marker="o", label=label, color=color)
        axes[1].plot(part["pilot_budget"], part["conformal_quantile"], marker="o", label=label, color=color)
    axes[0].axhline(0.90, color="#555555", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("Held-out lower-bound coverage")
    axes[1].set_ylabel("Conformal quantile")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=8)
        ax.set_xlabel("Pilot labels")
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Coverage")
    axes[1].set_title("Residual bound")
    axes[0].legend(frameon=False)
    path = Path(figures_dir) / "figure8_calibration_reliability.png"
    _finish(fig, path)
    return path


def figure9_near_oracle_ablation(gap: pd.DataFrame, figures_dir: str | Path) -> Path:
    rows = gap[
        (gap["experiment"].isin(["controlled_pilot_repair", "near_oracle_ablation"]))
        & (gap["pilot_budget"].isin([32, 128, 512]))
    ].copy()
    rows["label"] = rows["repair_model"] + " / " + rows["pilot_budget"].astype(int).astype(str)
    rows = rows.sort_values(["controlled_upper_bound", "repair_model", "pilot_budget"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#2ca02c" if not upper else "#111111" for upper in rows["controlled_upper_bound"]]
    ax.bar(np.arange(len(rows)), rows["gap_closed_clamped"], color=colors)
    ax.axhline(0.95, color="#555555", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xticks(np.arange(len(rows)), labels=rows["label"], rotation=25, ha="right")
    ax.set_ylabel("Oracle gap closed")
    ax.set_ylim(0.0, 1.08)
    ax.set_title("Near-oracle upper-bound ablation")
    ax.grid(True, axis="y", alpha=0.25)
    path = Path(figures_dir) / "figure9_near_oracle_ablation.png"
    _finish(fig, path)
    return path


def write_all_figures(results_dir: str | Path, figures_dir: str | Path) -> list[Path]:
    results_dir = Path(results_dir)
    metrics = pd.read_csv(results_dir / "tables" / "main_metrics.csv")
    grid = pd.read_csv(results_dir / "tables" / "denoising_grid.csv")
    validation = pd.read_csv(results_dir / "tables" / "exact_law_validation.csv")
    gap = pd.read_csv(results_dir / "tables" / "gap_closure_by_budget.csv")
    adaptive = pd.read_csv(results_dir / "tables" / "adaptive_n_metrics.csv")
    calibration = pd.read_csv(results_dir / "tables" / "calibration_diagnostics.csv")
    return [
        figure1_tail_hallucination(metrics, figures_dir),
        figure2_repair_comparison(metrics, figures_dir),
        figure3_tail_diagnostics(metrics, figures_dir),
        figure4_denoising_vs_selection(grid, figures_dir),
        figure5_exact_law_validation(validation, figures_dir),
        figure6_pilot_repair_gap_closure(gap, figures_dir),
        figure7_adaptive_n_gate(adaptive, figures_dir),
        figure8_calibration_reliability(calibration, figures_dir),
        figure9_near_oracle_ablation(gap, figures_dir),
    ]
