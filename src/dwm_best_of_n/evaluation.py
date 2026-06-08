"""Evaluation harness and metrics for Best-of-N diffusion-world-model selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .analytic_generators import CandidateBatch, generate_analytic_batch
from .scorers import score_candidates
from .theory import finite_best_of_n_law, tail_diagnostics
from .toy_world import ToyWorld


N_VALUES = (1, 2, 4, 8, 16, 32, 64)
GATE_DECISIONS = ("allow_high_n", "stop_early", "collect_pilot_labels", "block_high_n")
GATE_REASONS = (
    "utility_improves_with_confidence",
    "high_n_regret_detected",
    "pilot_labels_needed",
    "tail_rank_failure",
    "hidden_mode_unidentifiable",
    "oracle_baseline",
)


def ci95(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return 0.0
    return float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))


def deployment_gate(
    high_n_regret: float,
    upper_tail_rank_correlation: float,
    imagined_real_tail_gap: float,
    oracle_gap: float = 0.0,
) -> str:
    return deployment_gate_with_reason(
        high_n_regret,
        upper_tail_rank_correlation,
        imagined_real_tail_gap,
        oracle_gap,
    )[0]


def deployment_gate_with_reason(
    high_n_regret: float,
    upper_tail_rank_correlation: float,
    imagined_real_tail_gap: float,
    oracle_gap: float = 0.0,
) -> tuple[str, str]:
    if high_n_regret > 0.18 and upper_tail_rank_correlation < 0.18:
        return "block_high_n", "tail_rank_failure"
    if high_n_regret > 0.06 or (imagined_real_tail_gap > 0.9 and oracle_gap > 0.10):
        return "collect_pilot_labels", "pilot_labels_needed"
    if high_n_regret > 0.015:
        return "stop_early", "high_n_regret_detected"
    return "allow_high_n", "utility_improves_with_confidence"


def _selected_index(scores: np.ndarray, n: int) -> int:
    local = np.asarray(scores[:n], dtype=float)
    return int(np.flatnonzero(local == local.max())[0])


def candidate_diversity(batch: CandidateBatch, n: int) -> float:
    finals = batch.future_states[:n, -1, :]
    if n < 2:
        return 0.0
    diffs = finals[:, None, :] - finals[None, :, :]
    pairwise = np.linalg.norm(diffs, axis=2)
    mode_div = len(set(map(str, batch.modes[:n]))) / max(n, 1)
    return float(pairwise[np.triu_indices(n, k=1)].mean() + 0.15 * mode_div)


def evaluate_batches(
    batches: list[CandidateBatch],
    ns: Iterable[int] = N_VALUES,
    scorer: str = "raw",
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for n in ns:
        selected_scores = []
        selected_imagined = []
        selected_real = []
        selected_oracle = []
        diversity = []
        tail_corr = []
        exact_errors = []
        tail_gaps = []
        for batch in batches:
            scores = score_candidates(batch, scorer=scorer, seed=seed + batch.condition_id + int(n))
            idx = _selected_index(scores, int(n))
            oracle_idx = int(np.argmax(batch.real_utility[: int(n)]))
            selected_scores.append(float(scores[idx]))
            selected_imagined.append(float(batch.imagined_score[idx]))
            selected_real.append(float(batch.real_utility[idx]))
            selected_oracle.append(float(batch.real_utility[oracle_idx]))
            diversity.append(candidate_diversity(batch, int(n)))
            diag = tail_diagnostics(scores[: int(n)], batch.real_utility[: int(n)], quantile=0.75)
            tail_corr.append(diag["upper_tail_rank_correlation"])
            tail_gaps.append(diag["imagined_real_tail_gap"])
            law = finite_best_of_n_law(scores, batch.real_utility, int(n))
            exact_errors.append(abs(law.expected_utility - float(batch.real_utility[idx])))
        rows.append(
            {
                "N": int(n),
                "scorer": scorer,
                "selected_score_mean": float(np.mean(selected_scores)),
                "selected_imagined_score_mean": float(np.mean(selected_imagined)),
                "selected_real_utility_mean": float(np.mean(selected_real)),
                "oracle_real_utility_mean": float(np.mean(selected_oracle)),
                "generated_future_diversity": float(np.mean(diversity)),
                "upper_tail_rank_correlation": float(np.nanmean(tail_corr)),
                "imagined_real_tail_gap": float(np.nanmean(tail_gaps)),
                "oracle_gap": float(np.mean(selected_oracle) - np.mean(selected_real)),
                "exact_law_prediction_error": float(np.mean(exact_errors)),
            }
        )
    df = pd.DataFrame(rows)
    running_best = df["selected_real_utility_mean"].cummax()
    df["high_n_regret"] = np.maximum(0.0, running_best - df["selected_real_utility_mean"])
    max_row = df.loc[df["N"] == max(ns)].iloc[0]
    decision, reason = deployment_gate_with_reason(
        float(max_row["high_n_regret"]),
        float(max_row["upper_tail_rank_correlation"]),
        float(max_row["imagined_real_tail_gap"]),
        float(max_row["oracle_gap"]),
    )
    df["deployment_gate"] = decision
    df["gate_reason"] = reason
    return df


def aggregate_seed_metrics(seed_rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "generator", "scorer", "N"]
    metric_cols = [
        "selected_score_mean",
        "selected_imagined_score_mean",
        "selected_real_utility_mean",
        "oracle_real_utility_mean",
        "generated_future_diversity",
        "upper_tail_rank_correlation",
        "imagined_real_tail_gap",
        "oracle_gap",
        "exact_law_prediction_error",
        "high_n_regret",
    ]
    rows = []
    for keys, group in seed_rows.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        for col in metric_cols:
            row[col] = float(group[col].mean())
        row["selected_real_utility_ci95"] = ci95(group["selected_real_utility_mean"])
        row["selected_imagined_score_ci95"] = ci95(group["selected_imagined_score_mean"])
        row["deployment_gate"] = "collect_pilot_labels"
        row["gate_reason"] = "pilot_labels_needed"
        if "denoising_steps" in group:
            row["denoising_steps"] = int(group["denoising_steps"].iloc[0])
        rows.append(row)
    df = pd.DataFrame(rows)
    for keys, group in df.groupby(["experiment", "generator", "scorer"], sort=False):
        max_n = int(group["N"].max())
        high = group[group["N"] == max_n].iloc[0]
        decision, reason = deployment_gate_with_reason(
            float(high["high_n_regret"]),
            float(high["upper_tail_rank_correlation"]),
            float(high["imagined_real_tail_gap"]),
            float(high["oracle_gap"]),
        )
        mask = (
            (df["experiment"] == keys[0])
            & (df["generator"] == keys[1])
            & (df["scorer"] == keys[2])
        )
        df.loc[mask, "deployment_gate"] = decision
        df.loc[mask, "gate_reason"] = reason
    return df


def make_analytic_batches(
    variant: str,
    seed: int,
    n_conditions: int,
    max_n: int,
    denoising_steps: int = 8,
) -> list[CandidateBatch]:
    world = ToyWorld()
    rng = np.random.default_rng(seed)
    batches: list[CandidateBatch] = []
    for i in range(n_conditions):
        cond = world.sample_condition(rng, condition_id=i)
        batches.append(
            generate_analytic_batch(
                variant,
                world,
                cond,
                max_n,
                seed=seed * 100_003 + i * 97,
                denoising_steps=denoising_steps,
            )
        )
    return batches


def evaluate_analytic_variant(
    experiment: str,
    variant: str,
    scorers: Iterable[str],
    seeds: Iterable[int],
    n_conditions: int,
    ns: Iterable[int] = N_VALUES,
    denoising_steps: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_n = max(ns)
    seed_frames = []
    for seed in seeds:
        batches = make_analytic_batches(variant, seed, n_conditions, max_n, denoising_steps)
        for scorer in scorers:
            frame = evaluate_batches(batches, ns=ns, scorer=scorer, seed=seed)
            frame["seed"] = seed
            frame["experiment"] = experiment
            frame["generator"] = variant
            frame["denoising_steps"] = denoising_steps
            seed_frames.append(frame)
    seed_rows = pd.concat(seed_frames, ignore_index=True)
    return aggregate_seed_metrics(seed_rows), seed_rows


def evaluate_denoising_grid(
    seeds: Iterable[int],
    n_conditions: int,
    denoising_steps_grid: Iterable[int] = (2, 4, 8, 16),
    ns: Iterable[int] = N_VALUES,
) -> pd.DataFrame:
    frames = []
    for steps in denoising_steps_grid:
        agg, _ = evaluate_analytic_variant(
            "denoising_vs_selection",
            "optimistic",
            ["raw"],
            seeds,
            n_conditions,
            ns=ns,
            denoising_steps=int(steps),
        )
        agg["denoising_steps"] = int(steps)
        agg["runtime_units"] = agg["N"] * int(steps) * n_conditions * len(tuple(seeds))
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
