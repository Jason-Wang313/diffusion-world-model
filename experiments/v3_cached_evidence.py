from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "results" / "v3_cached_evidence"
FIG_OUT = OUT / "figures"
PAPER_FIG_OUT = ROOT / "figures" / "v3"
MACROS = ROOT / "paper_iclr" / "v3_results_macros.tex"
PDF_METADATA = {
    "Creator": "experiments/v3_cached_evidence.py",
    "CreationDate": None,
    "ModDate": None,
}


def read_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    return "NA"


def macro_line(name: str, value: float | int | str, digits: int = 3) -> str:
    rendered = fmt(value, digits) if isinstance(value, (float, np.floating)) else str(value)
    return f"\\newcommand{{\\{name}}}{{{rendered}}}\n"


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, metadata=PDF_METADATA)
    plt.close(fig)


def copy_figures() -> None:
    PAPER_FIG_OUT.mkdir(parents=True, exist_ok=True)
    for figure in sorted(FIG_OUT.glob("*.pdf")):
        (PAPER_FIG_OUT / figure.name).write_bytes(figure.read_bytes())


def row_at(df: pd.DataFrame, **filters: Any) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        mask &= df[key] == value
    rows = df[mask]
    if rows.empty:
        raise KeyError(filters)
    return rows.iloc[0]


def artifact_inventory() -> pd.DataFrame:
    rows = []
    for root_name in ["figures", "results", "paper_iclr", "src", "tests", "docs"]:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if OUT in path.parents:
                continue
            if path.is_file() and "__pycache__" not in path.parts:
                rows.append(
                    {
                        "root": root_name,
                        "suffix": path.suffix.lower() or "none",
                        "path": path.relative_to(ROOT).as_posix(),
                        "bytes": path.stat().st_size,
                    }
                )
    return pd.DataFrame(rows)


def write_claim_inventory(claims_payload: dict[str, Any]) -> pd.DataFrame:
    claims = pd.DataFrame(claims_payload["claims"])
    counts = claims["status"].value_counts().to_dict()
    rows = [
        {"status": "SUPPORTED", "count": int(counts.get("SUPPORTED", 0))},
        {"status": "PARTIAL", "count": int(counts.get("PARTIAL", 0))},
        {"status": "UNSUPPORTED", "count": int(counts.get("UNSUPPORTED", 0))},
    ]
    return pd.DataFrame(rows)


def build_outputs() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    main = read_csv("main_metrics.csv")
    learned = read_csv("learned_metrics.csv")
    seed = read_csv("seed_metrics.csv")
    denoising = read_csv("denoising_grid.csv")
    gap = read_csv("gap_closure_by_budget.csv")
    adaptive = read_csv("adaptive_n_metrics.csv")
    calibration = read_csv("calibration_diagnostics.csv")
    generalization = read_csv("learned_generalization_metrics.csv")
    exact = read_csv("exact_law_validation.csv")
    training = read_csv("learned_training_curve.csv")
    claims_payload = read_json("results/claims_status.json")
    run_summary = read_json("results/run_summary.json")

    controlled_n64 = main[
        (main["N"] == 64)
        & (main["experiment"].isin(["controlled", "learned_diffusion_world_model"]))
        & (main["scorer"].isin(["raw", "calibrated"]))
    ].copy()
    repair_n64 = main[(main["N"] == 64) & (main["experiment"] == "repair")].copy()
    n64 = pd.concat([controlled_n64, repair_n64], ignore_index=True, sort=False)
    n64 = n64[
        [
            "experiment",
            "generator",
            "scorer",
            "selected_imagined_score_mean",
            "selected_real_utility_mean",
            "oracle_real_utility_mean",
            "imagined_real_tail_gap",
            "oracle_gap",
            "high_n_regret",
            "deployment_gate",
            "gate_reason",
        ]
    ].sort_values(["experiment", "generator", "scorer"])
    n64.to_csv(OUT / "v3_n64_tail_failures.csv", index=False)

    seed_n64 = seed[(seed["N"] == 64) & (seed["scorer"] == "raw")].copy()
    seed_robustness = (
        seed_n64.groupby(["experiment", "generator"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            selected_real_mean=("selected_real_utility_mean", "mean"),
            selected_real_std=("selected_real_utility_mean", "std"),
            selected_real_min=("selected_real_utility_mean", "min"),
            selected_real_max=("selected_real_utility_mean", "max"),
            tail_gap_mean=("imagined_real_tail_gap", "mean"),
            tail_gap_max=("imagined_real_tail_gap", "max"),
            regret_mean=("high_n_regret", "mean"),
            regret_max=("high_n_regret", "max"),
        )
        .sort_values(["experiment", "generator"])
    )
    seed_robustness.to_csv(OUT / "v3_seed_robustness.csv", index=False)

    gap_sorted = gap.sort_values(["experiment", "repair_model", "pilot_budget"]).copy()
    gap_sorted.to_csv(OUT / "v3_repair_budget.csv", index=False)

    denoising_out = denoising.sort_values(["denoising_steps", "N"]).copy()
    denoising_out.to_csv(OUT / "v3_denoising_grid.csv", index=False)

    calibration_sorted = calibration.sort_values(["experiment", "repair_model", "pilot_budget"]).copy()
    calibration_sorted.to_csv(OUT / "v3_calibration_diagnostics.csv", index=False)

    claim_inventory = write_claim_inventory(claims_payload)
    claim_inventory.to_csv(OUT / "v3_claim_inventory.csv", index=False)

    inventory = artifact_inventory()
    inventory_summary = (
        inventory.groupby(["root", "suffix"], as_index=False)
        .agg(files=("path", "count"), bytes=("bytes", "sum"))
        .sort_values(["root", "suffix"])
    )
    inventory_summary.to_csv(OUT / "v3_artifact_inventory.csv", index=False)

    # Figure 1: N=64 selected-tail failure landscape.
    failure_plot = n64[n64["scorer"].isin(["raw", "calibrated"])].copy()
    failure_plot["label"] = failure_plot["generator"] + " / " + failure_plot["scorer"]
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    ax.barh(failure_plot["label"], failure_plot["imagined_real_tail_gap"], color="#9b2c2c")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("imagined-real tail gap at N=64")
    ax.set_title("Selected-tail hallucination severity")
    savefig(fig, FIG_OUT / "v3_tail_failure_landscape.pdf")

    # Figure 2: seed robustness for raw N=64 rows.
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    labels = seed_robustness["generator"] + " / " + seed_robustness["experiment"]
    y = np.arange(len(seed_robustness))
    means = seed_robustness["selected_real_mean"].to_numpy(float)
    lower = means - seed_robustness["selected_real_min"].to_numpy(float)
    upper = seed_robustness["selected_real_max"].to_numpy(float) - means
    ax.errorbar(means, y, xerr=[lower, upper], fmt="o", color="#225ea8", capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("selected real utility at N=64 across seeds")
    ax.set_title("Seed robustness of raw selected tails")
    savefig(fig, FIG_OUT / "v3_seed_robustness.pdf")

    # Figure 3: repair budget sensitivity.
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for (experiment, repair_model), rows in gap_sorted.groupby(["experiment", "repair_model"]):
        deployable = bool(rows["deployable_repair"].iloc[0])
        style = "-" if deployable else "--"
        ax.plot(
            rows["pilot_budget"],
            rows["gap_closed_clamped"],
            marker="o",
            linestyle=style,
            label=f"{experiment}: {repair_model}",
        )
    ax.set_xscale("symlog", linthresh=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("pilot labels")
    ax.set_ylabel("gap closed")
    ax.set_title("Pilot repair and upper-bound ablations")
    ax.legend(fontsize=7)
    savefig(fig, FIG_OUT / "v3_repair_budget_curve.pdf")

    # Figure 4: denoising-versus-selection heatmap.
    pivot = denoising_out.pivot(index="denoising_steps", columns="N", values="selected_real_utility_mean")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(i)) for i in pivot.index])
    ax.set_xlabel("selection budget N")
    ax.set_ylabel("denoising steps K")
    ax.set_title("Selected real utility under denoising and selection budgets")
    fig.colorbar(im, ax=ax, label="selected real utility")
    savefig(fig, FIG_OUT / "v3_denoising_selection_heatmap.pdf")

    # Figure 5: calibration coverage.
    cal_plot = calibration_sorted[calibration_sorted["pilot_budget"] > 0].copy()
    cal_plot["label"] = cal_plot["experiment"] + " / " + cal_plot["repair_model"] + " / " + cal_plot["pilot_budget"].astype(str)
    fig, ax = plt.subplots(figsize=(8.2, max(3.8, 0.35 * len(cal_plot))))
    ax.barh(cal_plot["label"], cal_plot["eval_lower_bound_coverage"].fillna(0.0), color="#238b45")
    ax.axvline(0.9, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("held-out lower-bound coverage")
    ax.set_title("Calibration coverage by repair setting")
    savefig(fig, FIG_OUT / "v3_calibration_coverage.pdf")

    # Figure 6: claim and artifact inventory.
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.8))
    axes[0].bar(claim_inventory["status"], claim_inventory["count"], color=["#225ea8", "#fd8d3c", "#bdbdbd"])
    axes[0].set_title("Claim ledger")
    axes[0].set_ylabel("claims")
    suffix_counts = inventory.groupby("suffix")["path"].count().sort_values(ascending=False).head(6)
    axes[1].bar(suffix_counts.index, suffix_counts.values, color="#756bb1")
    axes[1].set_title("Artifact suffixes")
    axes[1].set_ylabel("files")
    savefig(fig, FIG_OUT / "v3_claim_artifact_inventory.pdf")

    copy_figures()

    optimistic = row_at(main, experiment="controlled", generator="optimistic", scorer="raw", N=64)
    mode_collapsed = row_at(main, experiment="controlled", generator="mode_collapsed", scorer="raw", N=64)
    plausible = row_at(main, experiment="controlled", generator="plausibility_biased", scorer="raw", N=64)
    good = row_at(main, experiment="controlled", generator="good", scorer="raw", N=64)
    learned_raw = row_at(learned, scorer="raw", N=64)
    learned_cal = row_at(learned, scorer="calibrated", N=64)
    controlled_gap32 = row_at(gap, experiment="controlled_pilot_repair", repair_model="pilot_lcb", pilot_budget=32)
    controlled_gap128 = row_at(gap, experiment="controlled_pilot_repair", repair_model="pilot_lcb", pilot_budget=128)
    learned_gap32 = row_at(gap, experiment="learned_pilot_repair", repair_model="pilot_lcb", pilot_budget=32)
    oracle_gap = row_at(gap, experiment="near_oracle_ablation", repair_model="repair_oracle_features", pilot_budget=128)
    learned_test = row_at(generalization, split="test")
    learned_selection = row_at(generalization, split="heldout_selection")
    exact_max = float(exact["abs_error"].max())
    training_first = float(training[(training["ensemble_id"] == 0) & (training["epoch"] == 1)].iloc[0]["loss"])
    training_last = float(training[(training["ensemble_id"] == 0) & (training["epoch"] == training["epoch"].max())].iloc[0]["loss"])
    claims = claims_payload["claims"]
    supported = sum(1 for claim in claims if claim["status"] == "SUPPORTED")
    partial = sum(1 for claim in claims if claim["status"] == "PARTIAL")
    unsupported = sum(1 for claim in claims if claim["status"] == "UNSUPPORTED")

    summary = {
        "supported_claims": supported,
        "partial_claims": partial,
        "unsupported_boundary_claims": unsupported,
        "artifact_files": int(len(inventory)),
        "result_table_files": int(len(list(TABLES.glob("*.csv")))),
        "v3_figure_files": int(len(list(FIG_OUT.glob("*.pdf")))),
        "seed_rows": int(len(seed)),
        "main_metric_rows": int(len(main)),
        "denoising_rows": int(len(denoising)),
        "pilot_repair_rows": int(len(gap)),
        "adaptive_gate_rows": int(len(adaptive)),
        "calibration_rows": int(len(calibration)),
        "run_elapsed_seconds": float(run_summary["elapsed_seconds"]),
        "run_seeds": int(len(run_summary["settings"]["seeds"])),
        "run_conditions": int(run_summary["settings"]["n_conditions"]),
        "law_trials": int(run_summary["settings"]["law_trials"]),
        "exact_law_max_abs_error": exact_max,
        "good_raw_n64_real": float(good["selected_real_utility_mean"]),
        "optimistic_raw_n64_imagined": float(optimistic["selected_imagined_score_mean"]),
        "optimistic_raw_n64_real": float(optimistic["selected_real_utility_mean"]),
        "optimistic_raw_n64_tail_gap": float(optimistic["imagined_real_tail_gap"]),
        "optimistic_raw_n64_high_regret": float(optimistic["high_n_regret"]),
        "mode_collapsed_n64_tail_gap": float(mode_collapsed["imagined_real_tail_gap"]),
        "plausibility_biased_n64_tail_gap": float(plausible["imagined_real_tail_gap"]),
        "learned_raw_n64_imagined": float(learned_raw["selected_imagined_score_mean"]),
        "learned_raw_n64_real": float(learned_raw["selected_real_utility_mean"]),
        "learned_raw_n64_tail_gap": float(learned_raw["imagined_real_tail_gap"]),
        "learned_raw_n64_high_regret": float(learned_raw["high_n_regret"]),
        "learned_cal_n64_real": float(learned_cal["selected_real_utility_mean"]),
        "controlled_gap_closed_budget32": float(controlled_gap32["gap_closed"]),
        "controlled_gap_closed_budget128": float(controlled_gap128["gap_closed"]),
        "learned_gap_closed_budget32": float(learned_gap32["gap_closed"]),
        "oracle_feature_gap_closed": float(oracle_gap["gap_closed"]),
        "learned_test_mse": float(learned_test["future_trajectory_mse"]),
        "learned_test_final_state_error": float(learned_test["final_state_error"]),
        "learned_heldout_rank_correlation": float(learned_selection["utility_rank_correlation"]),
        "learned_heldout_sample_diversity": float(learned_selection["sample_diversity"]),
        "learned_tail_calibration_error": float(learned_selection["selected_tail_calibration_error"]),
        "training_loss_first": training_first,
        "training_loss_last": training_last,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with MACROS.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by experiments/v3_cached_evidence.py\n")
        handle.write(macro_line("VThreeDWMSupportedClaims", supported))
        handle.write(macro_line("VThreeDWMPartialClaims", partial))
        handle.write(macro_line("VThreeDWMUnsupportedBoundaryClaims", unsupported))
        handle.write(macro_line("VThreeDWMArtifactFiles", summary["artifact_files"]))
        handle.write(macro_line("VThreeDWMResultTables", summary["result_table_files"]))
        handle.write(macro_line("VThreeDWMSeedRows", summary["seed_rows"]))
        handle.write(macro_line("VThreeDWMDenoisingRows", summary["denoising_rows"]))
        handle.write(macro_line("VThreeDWMPilotRows", summary["pilot_repair_rows"]))
        handle.write(macro_line("VThreeDWMCalibrationRows", summary["calibration_rows"]))
        handle.write(macro_line("VThreeDWMRunSeconds", summary["run_elapsed_seconds"], 1))
        handle.write(macro_line("VThreeDWMRunSeeds", summary["run_seeds"]))
        handle.write(macro_line("VThreeDWMRunConditions", summary["run_conditions"]))
        handle.write(macro_line("VThreeDWMLawTrials", summary["law_trials"]))
        handle.write(macro_line("VThreeDWMExactLawMaxError", exact_max, 4))
        handle.write(macro_line("VThreeDWMGoodRawReal", summary["good_raw_n64_real"], 3))
        handle.write(macro_line("VThreeDWMOptimisticImagined", summary["optimistic_raw_n64_imagined"], 3))
        handle.write(macro_line("VThreeDWMOptimisticReal", summary["optimistic_raw_n64_real"], 3))
        handle.write(macro_line("VThreeDWMOptimisticTailGap", summary["optimistic_raw_n64_tail_gap"], 3))
        handle.write(macro_line("VThreeDWMOptimisticRegret", summary["optimistic_raw_n64_high_regret"], 3))
        handle.write(macro_line("VThreeDWMModeCollapsedTailGap", summary["mode_collapsed_n64_tail_gap"], 3))
        handle.write(macro_line("VThreeDWMPlausibilityTailGap", summary["plausibility_biased_n64_tail_gap"], 3))
        handle.write(macro_line("VThreeDWMLearnedRawImagined", summary["learned_raw_n64_imagined"], 3))
        handle.write(macro_line("VThreeDWMLearnedRawReal", summary["learned_raw_n64_real"], 3))
        handle.write(macro_line("VThreeDWMLearnedRawTailGap", summary["learned_raw_n64_tail_gap"], 3))
        handle.write(macro_line("VThreeDWMLearnedRawRegret", summary["learned_raw_n64_high_regret"], 3))
        handle.write(macro_line("VThreeDWMLearnedCalReal", summary["learned_cal_n64_real"], 3))
        handle.write(macro_line("VThreeDWMControlledGapBThirtyTwo", summary["controlled_gap_closed_budget32"], 3))
        handle.write(macro_line("VThreeDWMControlledGapBOneTwentyEight", summary["controlled_gap_closed_budget128"], 3))
        handle.write(macro_line("VThreeDWMLearnedGapBThirtyTwo", summary["learned_gap_closed_budget32"], 3))
        handle.write(macro_line("VThreeDWMOracleFeatureGap", summary["oracle_feature_gap_closed"], 3))
        handle.write(macro_line("VThreeDWMLearnedTestMSE", summary["learned_test_mse"], 3))
        handle.write(macro_line("VThreeDWMLearnedFinalError", summary["learned_test_final_state_error"], 3))
        handle.write(macro_line("VThreeDWMLearnedHeldoutRankCorr", summary["learned_heldout_rank_correlation"], 3))
        handle.write(macro_line("VThreeDWMLearnedHeldoutDiversity", summary["learned_heldout_sample_diversity"], 3))
        handle.write(macro_line("VThreeDWMLearnedTailCalibrationError", summary["learned_tail_calibration_error"], 3))
        handle.write(macro_line("VThreeDWMTrainingLossFirst", summary["training_loss_first"], 3))
        handle.write(macro_line("VThreeDWMTrainingLossLast", summary["training_loss_last"], 3))

    return summary


def main() -> None:
    summary = build_outputs()
    print(f"v3 cached evidence complete: {OUT}")
    print(
        "claims={supported} tables={tables} seed_rows={seed_rows} figures={figures}".format(
            supported=summary["supported_claims"],
            tables=summary["result_table_files"],
            seed_rows=summary["seed_rows"],
            figures=summary["v3_figure_files"],
        )
    )


if __name__ == "__main__":
    main()
