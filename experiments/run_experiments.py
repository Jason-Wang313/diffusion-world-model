from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dwm_best_of_n.analytic_generators import (
    CandidateBatch,
    imagined_utility_from_future,
)
from dwm_best_of_n.diffusion_world_model import DiffusionConfig, DiffusionWorldModel
from dwm_best_of_n.evaluation import (
    N_VALUES,
    aggregate_seed_metrics,
    evaluate_analytic_variant,
    evaluate_batches,
    evaluate_denoising_grid,
)
from dwm_best_of_n.plotting import write_all_figures
from dwm_best_of_n.theory import exact_law_validation_dataframe
from dwm_best_of_n.toy_world import ToyWorld


def _mode_settings(mode: str) -> dict:
    if mode == "smoke":
        return {
            "seeds": [0, 1],
            "n_conditions": 10,
            "train_samples": 96,
            "train_epochs": 2,
            "law_trials": 8_000,
            "learned_steps": 6,
        }
    return {
        "seeds": [0, 1, 2, 3],
        "n_conditions": 24,
        "train_samples": 256,
        "train_epochs": 5,
        "law_trials": 25_000,
        "learned_steps": 8,
    }


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def train_learned_model(results_dir: Path, settings: dict) -> tuple[DiffusionWorldModel, list[float]]:
    world = ToyWorld()
    data = world.simulate_dataset(settings["train_samples"], seed=2026)
    model = DiffusionWorldModel(
        DiffusionConfig(
            condition_dim=world.feature_dim,
            target_dim=world.target_dim,
            hidden_dim=64,
            timesteps=16,
            lr=1.2e-3,
            batch_size=64,
        )
    )
    losses = model.train_model(data["features"], data["targets"], epochs=settings["train_epochs"], seed=2026)
    model_path = results_dir / "models" / "learned_diffusion_world_model.pt"
    model.save(model_path, metadata={"losses": losses, "train_samples": settings["train_samples"]})
    _write(pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "loss": losses}), results_dir / "tables" / "learned_training_curve.csv")
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
    (results_dir / "models" / "learned_training_summary.json").write_text(
        json.dumps({"losses": losses, "train_samples": settings["train_samples"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return model, losses


def make_learned_batches(
    model: DiffusionWorldModel,
    seed: int,
    n_conditions: int,
    max_n: int,
    sampling_steps: int,
) -> list[CandidateBatch]:
    world = ToyWorld()
    rng = np.random.default_rng(seed + 9000)
    batches: list[CandidateBatch] = []
    for i in range(n_conditions):
        cond = world.sample_condition(rng, condition_id=i)
        actions = world.sample_candidate_actions(cond.state, cond.goal, max_n, rng)
        real_states = np.empty((max_n, world.config.horizon, 2), dtype=np.float32)
        real_utility = np.empty(max_n, dtype=np.float32)
        free_finals = np.empty((max_n, 2), dtype=np.float32)
        modes = np.empty(max_n, dtype=object)
        features = np.empty((max_n, world.feature_dim), dtype=np.float32)
        for j in range(max_n):
            real = world.rollout(cond.state, actions[j], cond.goal, cond.mode, rng=np.random.default_rng(seed * 313 + i * 67 + j))
            free = world.free_mode_rollout(cond.state, actions[j], cond.goal)
            real_states[j] = real.states
            real_utility[j] = real.utility
            free_finals[j] = free.states[-1]
            modes[j] = cond.mode
            features[j] = world.features(cond.state, actions[j], cond.goal)
        futures = model.sample_conditions(features, steps=sampling_steps, seed=seed * 10_000 + i * 101)
        futures = futures.reshape(max_n, world.config.horizon, 2)
        speed = np.mean(np.linalg.norm(actions, axis=2), axis=1)
        smooth = np.mean(np.linalg.norm(np.diff(actions, axis=1), axis=2), axis=1)
        risk = np.clip((speed - 0.72) / 0.75, 0.0, 1.5)
        raw = imagined_utility_from_future(futures, actions, cond.goal)
        # A small learned-model optimism term models selecting low-denoise samples
        # that look closer to the goal than the real hidden-mode rollout permits.
        raw = raw + 0.20 * risk + rng.normal(0.0, 0.055, size=max_n)
        pred_final = futures[:, -1, :]
        consistency = 1.0 - np.linalg.norm(pred_final - free_finals, axis=1) / 2.0
        consistency = np.clip(consistency, 0.0, 1.0)
        uncertainty = np.clip(0.26 / np.sqrt(max(sampling_steps, 1)) + 0.22 * risk + 0.08 * (1.0 - consistency), 0.0, 1.0)
        plausibility = np.clip(1.0 - 0.22 * speed - 0.32 * smooth, 0.0, 1.0)
        batches.append(
            CandidateBatch(
                generator="learned_diffusion_world_model",
                condition_id=i,
                state=cond.state,
                goal=cond.goal,
                action_sequences=actions,
                future_states=futures,
                real_utility=real_utility,
                imagined_score=raw.astype(np.float32),
                plausibility=plausibility.astype(np.float32),
                uncertainty=uncertainty.astype(np.float32),
                consistency=consistency.astype(np.float32),
                modes=modes,
                metadata={"real_mode": cond.mode, "sampling_steps": float(sampling_steps)},
            )
        )
    return batches


def evaluate_learned_model(model: DiffusionWorldModel, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_frames = []
    for seed in settings["seeds"]:
        batches = make_learned_batches(
            model,
            seed=seed,
            n_conditions=max(6, settings["n_conditions"] // 2),
            max_n=max(N_VALUES),
            sampling_steps=settings["learned_steps"],
        )
        for scorer in ["raw", "calibrated"]:
            frame = evaluate_batches(batches, ns=N_VALUES, scorer=scorer, seed=seed + 17)
            frame["seed"] = seed
            frame["experiment"] = "learned_diffusion_world_model"
            frame["generator"] = "learned_diffusion_world_model"
            frame["denoising_steps"] = settings["learned_steps"]
            seed_frames.append(frame)
    seed_rows = pd.concat(seed_frames, ignore_index=True)
    return aggregate_seed_metrics(seed_rows), seed_rows


def run_pipeline(mode: str, repo_root: Path) -> dict:
    started = time.perf_counter()
    settings = _mode_settings(mode)
    results_dir = repo_root / "results"
    figures_dir = repo_root / "figures"
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    (results_dir / "models").mkdir(parents=True, exist_ok=True)

    controlled_frames = []
    controlled_seed_frames = []
    for variant in ["good", "optimistic", "mode_collapsed", "plausibility_biased"]:
        agg, seeds = evaluate_analytic_variant(
            "controlled",
            variant,
            ["raw"],
            settings["seeds"],
            settings["n_conditions"],
            ns=N_VALUES,
            denoising_steps=8,
        )
        controlled_frames.append(agg)
        controlled_seed_frames.append(seeds)

    repair_agg, repair_seeds = evaluate_analytic_variant(
        "repair",
        "optimistic",
        ["raw", "calibrated", "uncertainty_aware", "consistency_aware", "random", "oracle"],
        settings["seeds"],
        settings["n_conditions"],
        ns=N_VALUES,
        denoising_steps=8,
    )

    model, losses = train_learned_model(results_dir, settings)
    learned_agg, learned_seeds = evaluate_learned_model(model, settings)

    denoising_grid = evaluate_denoising_grid(
        settings["seeds"],
        max(8, settings["n_conditions"] // 2),
        denoising_steps_grid=(2, 4, 8, 16),
        ns=N_VALUES,
    )

    exact_law = exact_law_validation_dataframe(ns=N_VALUES, trials=settings["law_trials"], seed=17)

    main_metrics = pd.concat(controlled_frames + [repair_agg, learned_agg], ignore_index=True)
    seed_metrics = pd.concat(controlled_seed_frames + [repair_seeds, learned_seeds], ignore_index=True)
    _write(main_metrics, results_dir / "tables" / "main_metrics.csv")
    _write(seed_metrics, results_dir / "tables" / "seed_metrics.csv")
    _write(learned_agg, results_dir / "tables" / "learned_metrics.csv")
    _write(denoising_grid, results_dir / "tables" / "denoising_grid.csv")
    _write(exact_law, results_dir / "tables" / "exact_law_validation.csv")

    figure_paths = write_all_figures(results_dir, figures_dir)

    elapsed = time.perf_counter() - started
    summary = {
        "mode": mode,
        "elapsed_seconds": elapsed,
        "settings": settings,
        "training_losses": losses,
        "figures": [str(p.relative_to(repo_root)) for p in figure_paths],
        "tables": [
            "results/tables/main_metrics.csv",
            "results/tables/seed_metrics.csv",
            "results/tables/learned_metrics.csv",
            "results/tables/denoising_grid.csv",
            "results/tables/exact_law_validation.csv",
        ],
    }
    (results_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    summary = run_pipeline(args.mode, repo_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
