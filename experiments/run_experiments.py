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
    candidate_diversity,
    evaluate_analytic_variant,
    evaluate_batches,
    evaluate_denoising_grid,
    make_analytic_batches,
)
from dwm_best_of_n.pilot_repair import PILOT_BUDGETS, run_pilot_repair_experiment
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


def _train_single_learned_model(settings: dict, seed: int) -> tuple[DiffusionWorldModel, list[float]]:
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
    losses = model.train_model(data["features"], data["targets"], epochs=settings["train_epochs"], seed=seed)
    return model, losses


def train_learned_models(results_dir: Path, settings: dict, ensemble_size: int = 3) -> tuple[list[DiffusionWorldModel], list[list[float]]]:
    models: list[DiffusionWorldModel] = []
    losses_by_model: list[list[float]] = []
    rows = []
    for ensemble_id in range(ensemble_size):
        model, losses = _train_single_learned_model(settings, seed=2026 + 37 * ensemble_id)
        models.append(model)
        losses_by_model.append(losses)
        for epoch, loss in enumerate(losses, start=1):
            rows.append({"ensemble_id": ensemble_id, "epoch": epoch, "loss": loss})
        member_path = results_dir / "models" / f"learned_diffusion_world_model_ensemble{ensemble_id}.pt"
        model.save(member_path, metadata={"losses": losses, "train_samples": settings["train_samples"], "ensemble_id": ensemble_id})
    losses = losses_by_model[0]
    model_path = results_dir / "models" / "learned_diffusion_world_model.pt"
    models[0].save(model_path, metadata={"losses": losses, "train_samples": settings["train_samples"], "ensemble_id": 0})
    _write(pd.DataFrame(rows), results_dir / "tables" / "learned_training_curve.csv")
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
    (results_dir / "models" / "learned_training_summary.json").write_text(
        json.dumps({"losses": losses, "losses_by_model": losses_by_model, "train_samples": settings["train_samples"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return models, losses_by_model


def make_learned_batches(
    models: DiffusionWorldModel | list[DiffusionWorldModel],
    seed: int,
    n_conditions: int,
    max_n: int,
    sampling_steps: int,
) -> list[CandidateBatch]:
    model_list = models if isinstance(models, list) else [models]
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
        ensemble_samples = []
        for member_id, model in enumerate(model_list):
            sample = model.sample_conditions(
                features,
                steps=sampling_steps,
                seed=seed * 10_000 + i * 101 + member_id * 997,
            )
            ensemble_samples.append(sample.reshape(max_n, world.config.horizon, 2))
        ensemble_arr = np.stack(ensemble_samples, axis=0)
        futures = ensemble_arr.mean(axis=0).astype(np.float32)
        disagreement = np.linalg.norm(ensemble_arr[:, :, -1, :].std(axis=0), axis=1)
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
        uncertainty = np.clip(
            0.26 / np.sqrt(max(sampling_steps, 1))
            + 0.22 * risk
            + 0.08 * (1.0 - consistency)
            + 0.32 * disagreement,
            0.0,
            1.4,
        )
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
                ensemble_disagreement=disagreement.astype(np.float32),
                metadata={"real_mode": cond.mode, "sampling_steps": float(sampling_steps)},
            )
        )
    return batches


def evaluate_learned_model(
    models: list[DiffusionWorldModel],
    settings: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, list[CandidateBatch]]]]:
    seed_frames = []
    batch_groups: list[tuple[int, list[CandidateBatch]]] = []
    for seed in settings["seeds"]:
        batches = make_learned_batches(
            models,
            seed=seed,
            n_conditions=max(6, settings["n_conditions"] // 2),
            max_n=max(N_VALUES),
            sampling_steps=settings["learned_steps"],
        )
        batch_groups.append((seed, batches))
        for scorer in ["raw", "calibrated"]:
            frame = evaluate_batches(batches, ns=N_VALUES, scorer=scorer, seed=seed + 17)
            frame["seed"] = seed
            frame["experiment"] = "learned_diffusion_world_model"
            frame["generator"] = "learned_diffusion_world_model"
            frame["denoising_steps"] = settings["learned_steps"]
            seed_frames.append(frame)
    seed_rows = pd.concat(seed_frames, ignore_index=True)
    return aggregate_seed_metrics(seed_rows), seed_rows, batch_groups


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def learned_generalization_metrics(
    models: list[DiffusionWorldModel],
    settings: dict,
    learned_batch_groups: list[tuple[int, list[CandidateBatch]]],
) -> pd.DataFrame:
    world = ToyWorld()
    rows = []
    split_specs = [
        ("train", 2026, min(settings["train_samples"], 160)),
        ("validation", 3031, max(48, settings["train_samples"] // 3)),
        ("test", 4042, max(48, settings["train_samples"] // 3)),
    ]
    for split, seed, samples in split_specs:
        data = world.simulate_dataset(samples, seed=seed)
        member_preds = []
        for member_id, model in enumerate(models):
            pred = model.sample_conditions(
                data["features"],
                steps=settings["learned_steps"],
                seed=seed * 19 + member_id * 101,
            )
            member_preds.append(pred)
        preds = np.stack(member_preds, axis=0)
        mean_pred = preds.mean(axis=0)
        target = data["targets"]
        mse = float(np.mean((mean_pred - target) ** 2))
        final_dim = world.config.horizon * 2
        final_pred = mean_pred[:, final_dim - 2 : final_dim]
        final_target = target[:, final_dim - 2 : final_dim]
        final_state_error = float(np.mean(np.linalg.norm(final_pred - final_target, axis=1)))
        ensemble_diversity = float(np.mean(np.linalg.norm(preds[:, :, final_dim - 2 : final_dim].std(axis=0), axis=1)))
        rows.append(
            {
                "split": split,
                "condition_split": split,
                "future_trajectory_mse": mse,
                "final_state_error": final_state_error,
                "utility_rank_correlation": np.nan,
                "selected_tail_calibration_error": np.nan,
                "sample_diversity": ensemble_diversity,
                "negative_log_proxy": float(np.log(mse + 1e-6)),
                "denoising_loss_proxy": mse,
                "ensemble_size": len(models),
                "heldout_conditions": 0,
            }
        )

    imagined = []
    real = []
    diversity = []
    disagreement = []
    for _, batches in learned_batch_groups:
        for batch in batches:
            imagined.extend(map(float, batch.imagined_score))
            real.extend(map(float, batch.real_utility))
            diversity.append(candidate_diversity(batch, len(batch)))
            if batch.ensemble_disagreement is not None:
                disagreement.extend(map(float, batch.ensemble_disagreement))
    imagined_arr = np.asarray(imagined, dtype=float)
    real_arr = np.asarray(real, dtype=float)
    if imagined_arr.size:
        threshold = np.quantile(imagined_arr, 0.75)
        tail = imagined_arr >= threshold
        rank_corr = _rank_correlation(imagined_arr, real_arr)
        tail_error = float(np.mean(np.abs(imagined_arr[tail] - real_arr[tail])))
    else:
        rank_corr = np.nan
        tail_error = np.nan
    rows.append(
        {
            "split": "heldout_selection",
            "condition_split": "heldout_by_seed_condition",
            "future_trajectory_mse": np.nan,
            "final_state_error": np.nan,
            "utility_rank_correlation": rank_corr,
            "selected_tail_calibration_error": tail_error,
            "sample_diversity": float(np.mean(diversity)) if diversity else np.nan,
            "negative_log_proxy": np.nan,
            "denoising_loss_proxy": np.nan,
            "ensemble_size": len(models),
            "heldout_conditions": int(sum(len(batches) for _, batches in learned_batch_groups)),
            "ensemble_disagreement_mean": float(np.mean(disagreement)) if disagreement else 0.0,
        }
    )
    return pd.DataFrame(rows)


def make_analytic_batch_groups(settings: dict, variant: str = "optimistic") -> list[tuple[int, list[CandidateBatch]]]:
    return [
        (
            seed,
            make_analytic_batches(
                variant,
                seed=seed,
                n_conditions=settings["n_conditions"],
                max_n=max(N_VALUES),
                denoising_steps=8,
            ),
        )
        for seed in settings["seeds"]
    ]


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

    models, losses_by_model = train_learned_models(results_dir, settings)
    learned_agg, learned_seeds, learned_batch_groups = evaluate_learned_model(models, settings)
    learned_generalization = learned_generalization_metrics(models, settings, learned_batch_groups)

    practical_groups = make_analytic_batch_groups(settings, variant="optimistic")
    controlled_pilot, controlled_gap, controlled_adaptive, controlled_calibration = run_pilot_repair_experiment(
        practical_groups,
        budgets=PILOT_BUDGETS,
        ns=N_VALUES,
        experiment="controlled_pilot_repair",
        generator="optimistic",
        repair_model="pilot_lcb",
        include_oracle_features=False,
        split_seed=41,
    )
    learned_pilot, learned_gap, learned_adaptive, learned_calibration = run_pilot_repair_experiment(
        learned_batch_groups,
        budgets=(0, 8, 32),
        ns=N_VALUES,
        experiment="learned_pilot_repair",
        generator="learned_diffusion_world_model",
        repair_model="pilot_lcb",
        include_oracle_features=False,
        split_seed=53,
    )
    oracle_pilot, oracle_gap, oracle_adaptive, oracle_calibration = run_pilot_repair_experiment(
        practical_groups,
        budgets=(128,),
        ns=N_VALUES,
        experiment="near_oracle_ablation",
        generator="optimistic",
        repair_model="repair_oracle_features",
        include_oracle_features=True,
        split_seed=41,
    )
    many_label_pilot, many_label_gap, many_label_adaptive, many_label_calibration = run_pilot_repair_experiment(
        practical_groups,
        budgets=(512,),
        ns=N_VALUES,
        experiment="near_oracle_ablation",
        generator="optimistic",
        repair_model="repair_many_pilot_labels",
        include_oracle_features=False,
        split_seed=41,
    )

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
    _write(
        pd.concat([controlled_pilot, learned_pilot, oracle_pilot, many_label_pilot], ignore_index=True),
        results_dir / "tables" / "pilot_repair_metrics.csv",
    )
    _write(
        pd.concat([controlled_gap, learned_gap, oracle_gap, many_label_gap], ignore_index=True),
        results_dir / "tables" / "gap_closure_by_budget.csv",
    )
    _write(
        pd.concat([controlled_adaptive, learned_adaptive, oracle_adaptive, many_label_adaptive], ignore_index=True),
        results_dir / "tables" / "adaptive_n_metrics.csv",
    )
    _write(
        pd.concat([controlled_calibration, learned_calibration, oracle_calibration, many_label_calibration], ignore_index=True),
        results_dir / "tables" / "calibration_diagnostics.csv",
    )
    _write(learned_generalization, results_dir / "tables" / "learned_generalization_metrics.csv")

    figure_paths = write_all_figures(results_dir, figures_dir)

    elapsed = time.perf_counter() - started
    summary = {
        "mode": mode,
        "elapsed_seconds": elapsed,
        "settings": settings,
        "training_losses": losses_by_model[0],
        "training_losses_by_model": losses_by_model,
        "figures": [str(p.relative_to(repo_root)) for p in figure_paths],
        "tables": [
            "results/tables/main_metrics.csv",
            "results/tables/seed_metrics.csv",
            "results/tables/learned_metrics.csv",
            "results/tables/denoising_grid.csv",
            "results/tables/exact_law_validation.csv",
            "results/tables/pilot_repair_metrics.csv",
            "results/tables/gap_closure_by_budget.csv",
            "results/tables/adaptive_n_metrics.csv",
            "results/tables/calibration_diagnostics.csv",
            "results/tables/learned_generalization_metrics.csv",
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
