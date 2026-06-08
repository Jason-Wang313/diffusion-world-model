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


def write_all_figures(results_dir: str | Path, figures_dir: str | Path) -> list[Path]:
    results_dir = Path(results_dir)
    metrics = pd.read_csv(results_dir / "tables" / "main_metrics.csv")
    grid = pd.read_csv(results_dir / "tables" / "denoising_grid.csv")
    validation = pd.read_csv(results_dir / "tables" / "exact_law_validation.csv")
    return [
        figure1_tail_hallucination(metrics, figures_dir),
        figure2_repair_comparison(metrics, figures_dir),
        figure3_tail_diagnostics(metrics, figures_dir),
        figure4_denoising_vs_selection(grid, figures_dir),
        figure5_exact_law_validation(validation, figures_dir),
    ]
