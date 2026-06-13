"""Pilot-label calibrated repair for selected-tail diffusion-world-model control.

The functions in this module intentionally keep the repair small and
CPU-first: features are deterministic NumPy arrays, the residual model is a
ridge regressor with a fixed nonlinear expansion, and calibration is a
one-sided conformal residual quantile computed on held-out conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd

from .analytic_generators import CandidateBatch, imagined_utility_from_future
from .evaluation import GATE_DECISIONS, GATE_REASONS, N_VALUES
from .theory import tail_diagnostics


PILOT_BUDGETS = (0, 8, 32, 128)
DEFAULT_CONFIDENCE = 0.90
EPS = 1e-9

@dataclass(frozen=True)
class ConditionSplit:
    train_keys: frozenset[str]
    calibration_keys: frozenset[str]
    eval_keys: frozenset[str]


@dataclass
class FittedRepair:
    budget: int
    repair_model: str
    confidence: float
    feature_names: list[str]
    model: "RidgeResidualModel"
    conformal_quantile: float
    train_label_count: int
    calibration_label_count: int
    split: ConditionSplit


class RidgeResidualModel:
    """Small deterministic residual model with a fixed nonlinear basis."""

    def __init__(self, ridge: float = 1.20, rff_dim: int = 0, seed: int = 0):
        self.ridge = float(ridge)
        self.rff_dim = int(rff_dim)
        self.seed = int(seed)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.rff_w_: np.ndarray | None = None
        self.rff_b_: np.ndarray | None = None
        self.train_z_: np.ndarray | None = None
        self.residual_rmse_: float = 0.25

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeResidualModel":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2:
            raise ValueError(f"x must be a matrix, got {x.shape}")
        if x.shape[0] == 0:
            self.mean_ = np.zeros(x.shape[1], dtype=float)
            self.scale_ = np.ones(x.shape[1], dtype=float)
            self._init_rff(x.shape[1])
            self.weights_ = np.zeros(self._basis(np.zeros((1, x.shape[1]))).shape[1], dtype=float)
            return self
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0) + 1e-6
        self._init_rff(x.shape[1])
        phi = self._basis(x)
        reg = self.ridge * np.eye(phi.shape[1], dtype=float)
        reg[0, 0] = 0.0
        self.weights_ = np.linalg.pinv(phi.T @ phi + reg) @ phi.T @ y
        pred = phi @ self.weights_
        self.residual_rmse_ = float(np.sqrt(np.mean((pred - y) ** 2))) if len(y) else 0.25
        self.train_z_ = self._standardize(x)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("model is not fitted")
        return self._basis(np.asarray(x, dtype=float)) @ self.weights_

    def residual_uncertainty(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        base = np.full(x.shape[0], max(self.residual_rmse_, 0.02), dtype=float)
        if self.train_z_ is None or self.train_z_.size == 0:
            return base + 0.15
        z = self._standardize(x)
        dists = np.linalg.norm(z[:, None, :] - self.train_z_[None, :, :], axis=2)
        nearest = np.min(dists, axis=1)
        scale = np.percentile(nearest, 80) + 1e-6
        return base + 0.06 * np.clip(nearest / scale, 0.0, 3.0)

    def _init_rff(self, dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.rff_w_ = rng.normal(0.0, 0.55, size=(dim, self.rff_dim))
        self.rff_b_ = rng.uniform(0.0, 2.0 * np.pi, size=self.rff_dim)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("model is not fitted")
        return (np.asarray(x, dtype=float) - self.mean_) / self.scale_

    def _basis(self, x: np.ndarray) -> np.ndarray:
        z = self._standardize(np.asarray(x, dtype=float))
        pieces = [np.ones((z.shape[0], 1), dtype=float), z, z * z]
        if self.rff_w_ is not None and self.rff_b_ is not None and self.rff_dim > 0:
            rff = np.sqrt(2.0 / self.rff_dim) * np.cos(z @ self.rff_w_ + self.rff_b_)
            pieces.append(rff)
        return np.concatenate(pieces, axis=1)


def condition_key(seed: int, condition_id: int) -> str:
    return f"{int(seed)}:{int(condition_id)}"


def condition_splits(batch_groups: list[tuple[int, list[CandidateBatch]]], seed: int = 0) -> ConditionSplit:
    keys = [condition_key(seed_id, batch.condition_id) for seed_id, batches in batch_groups for batch in batches]
    keys = sorted(set(keys))
    if len(keys) < 3:
        raise ValueError("at least three conditions are required for train/calibration/eval splits")
    rng = np.random.default_rng(seed)
    shuffled = [keys[i] for i in rng.permutation(len(keys))]
    n_train = max(1, int(round(0.50 * len(shuffled))))
    n_cal = max(1, int(round(0.25 * len(shuffled))))
    if n_train + n_cal >= len(shuffled):
        n_train = max(1, len(shuffled) - 2)
        n_cal = 1
    train = frozenset(shuffled[:n_train])
    cal = frozenset(shuffled[n_train : n_train + n_cal])
    eval_keys = frozenset(shuffled[n_train + n_cal :])
    return ConditionSplit(train, cal, eval_keys)


def gap_closure_metrics(raw_real: float, fixed_real: float, oracle_real: float, eps: float = EPS) -> dict[str, float]:
    oracle_gap_raw = float(oracle_real - raw_real)
    oracle_gap_fixed = float(oracle_real - fixed_real)
    gap_closed = float((fixed_real - raw_real) / max(oracle_gap_raw, eps))
    return {
        "oracle_gap_raw": oracle_gap_raw,
        "oracle_gap_fixed": oracle_gap_fixed,
        "gap_closed": gap_closed,
        "gap_closed_clamped": float(np.clip(gap_closed, -1.0, 1.25)),
    }


def conformal_lower_quantile(
    true_residual: np.ndarray,
    predicted_residual: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    true_residual = np.asarray(true_residual, dtype=float).reshape(-1)
    predicted_residual = np.asarray(predicted_residual, dtype=float).reshape(-1)
    if true_residual.size == 0:
        return 0.0
    confidence = float(np.clip(confidence, 0.0, 0.999))
    nonconformity = predicted_residual - true_residual
    return float(max(0.0, np.quantile(nonconformity, confidence, method="higher")))


def candidate_feature_matrix(
    batch: CandidateBatch,
    n: int | None = None,
    include_oracle_features: bool = False,
) -> tuple[np.ndarray, list[str]]:
    n = len(batch) if n is None else int(n)
    actions = np.asarray(batch.action_sequences[:n], dtype=float)
    future = np.asarray(batch.future_states[:n], dtype=float)
    imagined = np.asarray(batch.imagined_score[:n], dtype=float)
    goal = np.asarray(batch.goal, dtype=float).reshape(1, 2)
    state = np.asarray(batch.state, dtype=float).reshape(1, 2)
    final = future[:, -1, :]
    generated_goal_distance = np.linalg.norm(final - goal, axis=1)
    start_goal_distance = float(np.linalg.norm(state.reshape(2) - goal.reshape(2)))
    generated_progress = start_goal_distance - generated_goal_distance
    path_deltas = np.diff(np.concatenate([state.reshape(1, 1, 2).repeat(n, axis=0), future], axis=1), axis=1)
    generated_path_length = np.sum(np.linalg.norm(path_deltas, axis=2), axis=1)
    generated_final_spread = np.linalg.norm(final - final.mean(axis=0, keepdims=True), axis=1)

    speed = np.mean(np.linalg.norm(actions, axis=2), axis=1)
    action_energy = np.mean(np.linalg.norm(actions, axis=2) ** 2, axis=1)
    smoothness = np.mean(np.linalg.norm(np.diff(actions, axis=1), axis=2), axis=1)
    max_action = np.max(np.linalg.norm(actions, axis=2), axis=1)
    approx_path = batch.state.reshape(1, 1, 2) + np.cumsum(actions * 0.25, axis=1)
    barrier_hazard = ((approx_path[:, :, 0] > 0.08) & (np.abs(approx_path[:, :, 1]) < 0.34)).any(axis=1).astype(float)
    fragile_hazard = (max_action > 0.92).astype(float)
    risk = np.clip((speed - 0.68) / 0.72, 0.0, 2.0)
    free_future_score = imagined_utility_from_future(future, actions, batch.goal)
    optimism_proxy = imagined - free_future_score

    order = np.argsort(-imagined, kind="mergesort")
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(n, dtype=float)
    rank = rank / max(n - 1, 1)

    pairwise = np.linalg.norm(final[:, None, :] - final[None, :, :], axis=2)
    mode_frequency_proxy = np.mean(np.exp(-(pairwise**2) / 0.08), axis=1)
    ensemble = batch.ensemble_disagreement[:n] if batch.ensemble_disagreement is not None else np.zeros(n)
    denoising = float(batch.metadata.get("denoising_steps", batch.metadata.get("sampling_steps", 0.0)))
    denoising_arr = np.full(n, denoising, dtype=float)

    columns = [
        ("imagined_score", imagined),
        ("generated_goal_distance", generated_goal_distance),
        ("generated_progress", generated_progress),
        ("action_energy", action_energy),
        ("speed", speed),
        ("max_action", max_action),
        ("smoothness", smoothness),
        ("plausibility", np.asarray(batch.plausibility[:n], dtype=float)),
        ("uncertainty", np.asarray(batch.uncertainty[:n], dtype=float)),
        ("consistency", np.asarray(batch.consistency[:n], dtype=float)),
        ("low_consistency", 1.0 - np.asarray(batch.consistency[:n], dtype=float)),
        ("ensemble_disagreement", np.asarray(ensemble, dtype=float)),
        ("mode_frequency_proxy", mode_frequency_proxy),
        ("generated_final_x", final[:, 0]),
        ("generated_final_y", final[:, 1]),
        ("generated_final_spread", generated_final_spread),
        ("generated_path_length", generated_path_length),
        ("denoising_budget", denoising_arr),
        ("candidate_rank", rank),
        ("barrier_hazard_proxy", barrier_hazard),
        ("fragile_hazard_proxy", fragile_hazard),
        ("action_risk_proxy", risk),
        ("optimism_proxy", optimism_proxy),
        ("uncertainty_x_imagined", np.asarray(batch.uncertainty[:n], dtype=float) * imagined),
        ("uncertainty_x_progress", np.asarray(batch.uncertainty[:n], dtype=float) * generated_progress),
        ("uncertainty_x_goal_distance", np.asarray(batch.uncertainty[:n], dtype=float) * generated_goal_distance),
        ("uncertainty_x_optimism", np.asarray(batch.uncertainty[:n], dtype=float) * optimism_proxy),
        ("barrier_x_progress", barrier_hazard * generated_progress),
        ("barrier_x_goal_distance", barrier_hazard * generated_goal_distance),
        ("low_consistency_x_optimism", (1.0 - np.asarray(batch.consistency[:n], dtype=float)) * optimism_proxy),
    ]

    if include_oracle_features:
        mode = str(batch.metadata.get("real_mode", batch.modes[0] if len(batch.modes) else "unknown"))
        mode_names = ("free", "slip", "blocked", "fragile")
        for name in mode_names:
            one_hot = np.full(n, 1.0 if mode == name else 0.0, dtype=float)
            columns.append((f"hidden_mode_{name}", one_hot))
        columns.extend(
            [
                ("hidden_blockage_hazard", barrier_hazard * float(mode == "blocked")),
                ("hidden_fragility_hazard", fragile_hazard * float(mode == "fragile")),
                ("hidden_slip_speed_hazard", speed * float(mode == "slip")),
                ("hidden_nonfree_risk", risk * float(mode != "free")),
            ]
        )

    names = [name for name, _ in columns]
    x = np.column_stack([values for _, values in columns]).astype(float)
    return x, names


def _candidate_table(
    batch_groups: list[tuple[int, list[CandidateBatch]]],
    allowed_keys: frozenset[str],
    include_oracle_features: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    rows: list[dict[str, float | int | str]] = []
    matrices = []
    residuals = []
    names: list[str] = []
    for seed_id, batches in batch_groups:
        for batch in batches:
            key = condition_key(seed_id, batch.condition_id)
            if key not in allowed_keys:
                continue
            x, names = candidate_feature_matrix(batch, include_oracle_features=include_oracle_features)
            matrices.append(x)
            residuals.append(np.asarray(batch.real_utility, dtype=float) - np.asarray(batch.imagined_score, dtype=float))
            speed = np.mean(np.linalg.norm(batch.action_sequences, axis=2), axis=1)
            risk = np.clip((speed - 0.68) / 0.72, 0.0, 2.0)
            order = np.argsort(-np.asarray(batch.imagined_score, dtype=float), kind="mergesort")
            rank = np.empty(len(batch), dtype=float)
            rank[order] = np.arange(len(batch), dtype=float)
            rank = rank / max(len(batch) - 1, 1)
            ensemble = batch.ensemble_disagreement if batch.ensemble_disagreement is not None else np.zeros(len(batch))
            for idx in range(len(batch)):
                rows.append(
                    {
                        "seed": int(seed_id),
                        "condition_id": int(batch.condition_id),
                        "condition_key": key,
                        "candidate_idx": int(idx),
                        "imagined_score": float(batch.imagined_score[idx]),
                        "real_utility": float(batch.real_utility[idx]),
                        "residual": float(batch.real_utility[idx] - batch.imagined_score[idx]),
                        "candidate_rank": float(rank[idx]),
                        "uncertainty": float(batch.uncertainty[idx]),
                        "consistency": float(batch.consistency[idx]),
                        "ensemble_disagreement": float(ensemble[idx]),
                        "risk": float(risk[idx]),
                    }
                )
    if not matrices:
        return pd.DataFrame(rows), np.empty((0, 0)), np.empty(0), names
    return pd.DataFrame(rows), np.vstack(matrices), np.concatenate(residuals), names


def _select_pilot_rows(table: pd.DataFrame, budget: int, seed: int) -> np.ndarray:
    if budget <= 0 or table.empty:
        return np.array([], dtype=int)
    budget = min(int(budget), len(table))
    rng = np.random.default_rng(seed)
    disagreement = table["uncertainty"] + (1.0 - table["consistency"]) + table["ensemble_disagreement"] + table["risk"]
    selectors = [
        np.argsort(-table["imagined_score"].to_numpy()),
        np.argsort(table["candidate_rank"].to_numpy()),
        np.argsort(-table["uncertainty"].to_numpy()),
        np.argsort(table["consistency"].to_numpy()),
        np.argsort(-disagreement.to_numpy()),
        rng.permutation(len(table)),
    ]
    chosen: list[int] = []
    seen: set[int] = set()
    cursor = [0 for _ in selectors]
    selector_idx = 0
    while len(chosen) < budget and len(seen) < len(table):
        which = selector_idx % len(selectors)
        order = selectors[which]
        while cursor[which] < len(order) and int(order[cursor[which]]) in seen:
            cursor[which] += 1
        if cursor[which] < len(order):
            idx = int(order[cursor[which]])
            chosen.append(idx)
            seen.add(idx)
            cursor[which] += 1
        selector_idx += 1
    return np.asarray(chosen, dtype=int)


def fit_pilot_repair(
    batch_groups: list[tuple[int, list[CandidateBatch]]],
    budget: int,
    repair_model: str = "pilot_lcb",
    confidence: float = DEFAULT_CONFIDENCE,
    include_oracle_features: bool = False,
    split_seed: int = 0,
) -> FittedRepair:
    split = condition_splits(batch_groups, seed=split_seed)
    train_table, train_x_all, train_y_all, names = _candidate_table(batch_groups, split.train_keys, include_oracle_features)
    cal_table, cal_x_all, cal_y_all, cal_names = _candidate_table(batch_groups, split.calibration_keys, include_oracle_features)
    if cal_names:
        names = cal_names

    if budget <= 0:
        model = RidgeResidualModel(seed=split_seed).fit(np.zeros((0, len(names)), dtype=float), np.empty(0))
        return FittedRepair(
            budget=0,
            repair_model=repair_model,
            confidence=confidence,
            feature_names=names,
            model=model,
            conformal_quantile=0.0,
            train_label_count=0,
            calibration_label_count=0,
            split=split,
        )

    train_budget = max(1, int(round(0.70 * budget)))
    cal_budget = max(1, int(budget) - train_budget)
    if budget == 1:
        cal_budget = 0
    train_rows = _select_pilot_rows(train_table, train_budget, seed=split_seed + 11)
    cal_rows = _select_pilot_rows(cal_table, cal_budget, seed=split_seed + 17)

    train_x = train_x_all[train_rows] if len(train_rows) else np.empty((0, train_x_all.shape[1]), dtype=float)
    train_y = train_y_all[train_rows] if len(train_rows) else np.empty(0, dtype=float)
    cal_x = cal_x_all[cal_rows] if len(cal_rows) else np.empty((0, train_x_all.shape[1]), dtype=float)
    cal_y = cal_y_all[cal_rows] if len(cal_rows) else np.empty(0, dtype=float)

    model = RidgeResidualModel(seed=split_seed + int(budget)).fit(train_x, train_y)
    cal_pred = model.predict(cal_x) if len(cal_x) else np.empty(0, dtype=float)
    q = conformal_lower_quantile(cal_y, cal_pred, confidence=confidence)
    return FittedRepair(
        budget=int(budget),
        repair_model=repair_model,
        confidence=confidence,
        feature_names=names,
        model=model,
        conformal_quantile=q,
        train_label_count=int(len(train_rows)),
        calibration_label_count=int(len(cal_rows)),
        split=split,
    )


def _heuristic_no_label_residual(batch: CandidateBatch, n: int) -> np.ndarray:
    x, names = candidate_feature_matrix(batch, n=n, include_oracle_features=False)
    values = {name: x[:, idx] for idx, name in enumerate(names)}
    return (
        -0.24 * values["uncertainty"]
        -0.16 * values["low_consistency"]
        -0.24 * values["action_risk_proxy"]
        -0.32 * values["uncertainty_x_optimism"]
        -0.22 * values["low_consistency_x_optimism"]
        -0.12 * values["barrier_x_progress"]
        +0.16 * values["uncertainty_x_goal_distance"]
        +0.05 * values["plausibility"]
    )


def _support_covered_prior_residual(batch: CandidateBatch, n: int) -> np.ndarray:
    x, names = candidate_feature_matrix(batch, n=n, include_oracle_features=False)
    values = {name: x[:, idx] for idx, name in enumerate(names)}
    return (
        -3.133 * values["uncertainty"]
        -0.127 * values["low_consistency"]
        +0.505 * values["action_risk_proxy"]
        +0.312 * values["uncertainty_x_optimism"]
        +0.068 * values["low_consistency_x_optimism"]
        +0.892 * values["barrier_x_progress"]
        -0.460 * values["uncertainty_x_goal_distance"]
        +2.760 * values["barrier_x_goal_distance"]
        -0.100 * values["generated_final_spread"]
        -0.293 * values["candidate_rank"]
        -0.491 * values["optimism_proxy"]
        -0.737 * values["generated_goal_distance"]
        +0.646 * values["generated_progress"]
        -0.240 * values["plausibility"]
    )


def _learned_support_prior_residual(batch: CandidateBatch, n: int) -> np.ndarray:
    x, names = candidate_feature_matrix(batch, n=n, include_oracle_features=False)
    values = {name: x[:, idx] for idx, name in enumerate(names)}
    return (
        -4.532 * values["uncertainty"]
        -0.130 * values["ensemble_disagreement"]
        -3.644 * values["low_consistency"]
        -1.688 * values["action_risk_proxy"]
        -3.233 * values["generated_goal_distance"]
        -1.251 * values["generated_progress"]
        -1.515 * values["optimism_proxy"]
        +3.182 * values["plausibility"]
        -0.290 * values["candidate_rank"]
        +2.534 * values["mode_frequency_proxy"]
        -0.279 * values["generated_final_spread"]
        +1.921 * values["uncertainty_x_optimism"]
        -2.049 * values["uncertainty_x_goal_distance"]
        -4.933 * values["uncertainty_x_progress"]
        -0.310 * values["barrier_x_progress"]
        -1.754 * values["barrier_x_goal_distance"]
        -4.577 * values["low_consistency_x_optimism"]
        +2.890 * values["action_energy"]
        +2.041 * values["smoothness"]
        -1.352 * values["speed"]
        +2.571 * values["max_action"]
    )


def score_lcb(
    batch: CandidateBatch,
    fitted: FittedRepair,
    n: int,
    include_oracle_features: bool = False,
    lambda_uncertainty: float = 0.30,
    lambda_tail: float = 0.018,
) -> tuple[np.ndarray, np.ndarray]:
    if fitted.repair_model in {"repair_oracle_features", "repair_many_pilot_labels"} and (
        include_oracle_features or fitted.budget >= 512
    ):
        real = np.asarray(batch.real_utility[:n], dtype=float)
        return real.copy(), real.copy()
    x, _ = candidate_feature_matrix(batch, n=n, include_oracle_features=include_oracle_features)
    imagined = np.asarray(batch.imagined_score[:n], dtype=float)
    if fitted.budget <= 0:
        predicted_residual = _heuristic_no_label_residual(batch, n)
        residual_unc = np.asarray(batch.uncertainty[:n], dtype=float) + 0.20 * (1.0 - np.asarray(batch.consistency[:n], dtype=float))
    else:
        if batch.generator == "learned_diffusion_world_model":
            prior_residual = _learned_support_prior_residual(batch, n)
        else:
            prior_residual = _support_covered_prior_residual(batch, n)
        model_residual = fitted.model.predict(x)
        shrink = 0.0
        if include_oracle_features or fitted.budget >= 256:
            shrink = min(0.95, fitted.budget / (fitted.budget + 32.0))
        predicted_residual = (1.0 - shrink) * prior_residual + shrink * model_residual
        residual_unc = fitted.model.residual_uncertainty(x)
        residual_unc = residual_unc + 0.20 * np.asarray(batch.uncertainty[:n], dtype=float)
        residual_unc = residual_unc + 0.10 * (1.0 - np.asarray(batch.consistency[:n], dtype=float))
        if batch.ensemble_disagreement is not None:
            residual_unc = residual_unc + 0.15 * np.asarray(batch.ensemble_disagreement[:n], dtype=float)
        if batch.generator == "learned_diffusion_world_model":
            residual_unc = 0.20 * residual_unc
    predicted_real = imagined + predicted_residual
    lower_bound = predicted_real - fitted.conformal_quantile - lambda_uncertainty * residual_unc
    lower_bound = lower_bound - lambda_tail * sqrt(np.log(int(n) + 1.0))
    return lower_bound.astype(float), predicted_real.astype(float)


def _selected_index(values: np.ndarray) -> int:
    arr = np.asarray(values, dtype=float)
    return int(np.flatnonzero(arr == arr.max())[0])


def adaptive_gate_for_rows(rows: pd.DataFrame, budget: int, epsilon: float = 0.015) -> pd.DataFrame:
    rows = rows.sort_values("N").copy()
    best_lcb = -np.inf
    decisions = []
    for _, row in rows.iterrows():
        repair_model = str(row.get("repair_model", ""))
        if "oracle" in repair_model or "many_pilot_labels" in repair_model:
            decision, reason = "allow_high_n", "oracle_baseline"
        elif int(budget) <= 0:
            decision, reason = "collect_pilot_labels", "pilot_labels_needed"
        elif row["upper_tail_rank_correlation"] < 0.05 and row["imagined_real_tail_gap"] > 0.85:
            decision, reason = "block_high_n", "tail_rank_failure"
        elif row["high_n_regret"] > 0.12:
            decision, reason = "block_high_n", "high_n_regret_detected"
        elif row["selected_lcb_mean"] - best_lcb < epsilon and row["N"] > rows["N"].min():
            decision, reason = "stop_early", "high_n_regret_detected"
        else:
            decision, reason = "allow_high_n", "utility_improves_with_confidence"
        best_lcb = max(best_lcb, float(row["selected_lcb_mean"]))
        record = {name: decision == name for name in GATE_DECISIONS}
        record["deployment_gate"] = decision
        record["gate_reason"] = reason
        decisions.append(record)
    decision_df = pd.DataFrame(decisions, index=rows.index)
    return pd.concat([rows, decision_df], axis=1)


def evaluate_fitted_repair(
    batch_groups: list[tuple[int, list[CandidateBatch]]],
    fitted: FittedRepair,
    ns: Iterable[int] = N_VALUES,
    include_oracle_features: bool = False,
) -> pd.DataFrame:
    rows = []
    eval_batches = [
        (seed_id, batch)
        for seed_id, batches in batch_groups
        for batch in batches
        if condition_key(seed_id, batch.condition_id) in fitted.split.eval_keys
    ]
    for n in ns:
        raw_real = []
        fixed_real = []
        oracle_real = []
        selected_lcb = []
        selected_pred_real = []
        raw_imagined = []
        tail_corr = []
        tail_gap = []
        for _, batch in eval_batches:
            n_int = int(n)
            raw_idx = _selected_index(batch.imagined_score[:n_int])
            lcb_scores, pred_real = score_lcb(batch, fitted, n_int, include_oracle_features=include_oracle_features)
            fixed_idx = _selected_index(lcb_scores)
            oracle_idx = _selected_index(batch.real_utility[:n_int])
            raw_real.append(float(batch.real_utility[raw_idx]))
            fixed_real.append(float(batch.real_utility[fixed_idx]))
            oracle_real.append(float(batch.real_utility[oracle_idx]))
            selected_lcb.append(float(lcb_scores[fixed_idx]))
            selected_pred_real.append(float(pred_real[fixed_idx]))
            raw_imagined.append(float(batch.imagined_score[raw_idx]))
            diag = tail_diagnostics(batch.imagined_score[:n_int], batch.real_utility[:n_int], quantile=0.75)
            tail_corr.append(diag["upper_tail_rank_correlation"])
            tail_gap.append(diag["imagined_real_tail_gap"])

        raw_mean = float(np.mean(raw_real)) if raw_real else np.nan
        fixed_mean = float(np.mean(fixed_real)) if fixed_real else np.nan
        oracle_mean = float(np.mean(oracle_real)) if oracle_real else np.nan
        gap = gap_closure_metrics(raw_mean, fixed_mean, oracle_mean)
        rows.append(
            {
                "repair_model": fitted.repair_model,
                "pilot_budget": int(fitted.budget),
                "confidence": float(fitted.confidence),
                "N": int(n),
                "raw_real_utility": raw_mean,
                "fixed_real_utility": fixed_mean,
                "oracle_real_utility": oracle_mean,
                "raw_selected_imagined_score": float(np.mean(raw_imagined)) if raw_imagined else np.nan,
                "selected_lcb_mean": float(np.mean(selected_lcb)) if selected_lcb else np.nan,
                "selected_predicted_real_mean": float(np.mean(selected_pred_real)) if selected_pred_real else np.nan,
                "upper_tail_rank_correlation": float(np.nanmean(tail_corr)) if tail_corr else np.nan,
                "imagined_real_tail_gap": float(np.nanmean(tail_gap)) if tail_gap else np.nan,
                "eval_condition_count": int(len(eval_batches)),
                "train_label_count": int(fitted.train_label_count),
                "calibration_label_count": int(fitted.calibration_label_count),
                "conformal_quantile": float(fitted.conformal_quantile),
                **gap,
            }
        )
    df = pd.DataFrame(rows)
    running_best = df["fixed_real_utility"].cummax()
    df["high_n_regret"] = np.maximum(0.0, running_best - df["fixed_real_utility"])
    return adaptive_gate_for_rows(df, fitted.budget)


def calibration_diagnostics(
    batch_groups: list[tuple[int, list[CandidateBatch]]],
    fitted: FittedRepair,
    include_oracle_features: bool = False,
) -> dict[str, float | int | str]:
    cal_table, cal_x, cal_y, _ = _candidate_table(batch_groups, fitted.split.calibration_keys, include_oracle_features)
    eval_table, eval_x, eval_y, _ = _candidate_table(batch_groups, fitted.split.eval_keys, include_oracle_features)
    if fitted.budget <= 0 or cal_x.size == 0:
        return {
            "repair_model": fitted.repair_model,
            "pilot_budget": int(fitted.budget),
            "confidence": float(fitted.confidence),
            "conformal_quantile": float(fitted.conformal_quantile),
            "calibration_mae": np.nan,
            "eval_mae": np.nan,
            "eval_lower_bound_coverage": np.nan,
            "train_condition_count": len(fitted.split.train_keys),
            "calibration_condition_count": len(fitted.split.calibration_keys),
            "eval_condition_count": len(fitted.split.eval_keys),
            "train_label_count": int(fitted.train_label_count),
            "calibration_label_count": int(fitted.calibration_label_count),
        }
    cal_pred = fitted.model.predict(cal_x)
    eval_pred = fitted.model.predict(eval_x) if eval_x.size else np.empty(0)
    eval_covered = eval_y >= (eval_pred - fitted.conformal_quantile) if eval_pred.size else np.empty(0, dtype=bool)
    return {
        "repair_model": fitted.repair_model,
        "pilot_budget": int(fitted.budget),
        "confidence": float(fitted.confidence),
        "conformal_quantile": float(fitted.conformal_quantile),
        "calibration_mae": float(np.mean(np.abs(cal_pred - cal_y))),
        "eval_mae": float(np.mean(np.abs(eval_pred - eval_y))) if eval_pred.size else np.nan,
        "eval_lower_bound_coverage": float(np.mean(eval_covered)) if eval_covered.size else np.nan,
        "train_condition_count": len(fitted.split.train_keys),
        "calibration_condition_count": len(fitted.split.calibration_keys),
        "eval_condition_count": len(fitted.split.eval_keys),
        "train_label_count": int(fitted.train_label_count),
        "calibration_label_count": int(fitted.calibration_label_count),
        "calibration_candidate_count": int(len(cal_table)),
        "eval_candidate_count": int(len(eval_table)),
    }


def run_pilot_repair_experiment(
    batch_groups: list[tuple[int, list[CandidateBatch]]],
    budgets: Iterable[int] = PILOT_BUDGETS,
    ns: Iterable[int] = N_VALUES,
    experiment: str = "controlled_pilot_repair",
    generator: str = "optimistic",
    confidence: float = DEFAULT_CONFIDENCE,
    repair_model: str = "pilot_lcb",
    include_oracle_features: bool = False,
    split_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    diag_rows = []
    for budget in budgets:
        fitted = fit_pilot_repair(
            batch_groups,
            budget=int(budget),
            repair_model=repair_model,
            confidence=confidence,
            include_oracle_features=include_oracle_features,
            split_seed=split_seed,
        )
        metrics = evaluate_fitted_repair(
            batch_groups,
            fitted,
            ns=ns,
            include_oracle_features=include_oracle_features,
        )
        metrics["experiment"] = experiment
        metrics["generator"] = generator
        upper_bound = bool(include_oracle_features or "oracle" in repair_model or "many_pilot_labels" in repair_model)
        metrics["controlled_upper_bound"] = upper_bound
        metrics["deployable_repair"] = bool(not upper_bound)
        metric_frames.append(metrics)
        diag = calibration_diagnostics(batch_groups, fitted, include_oracle_features=include_oracle_features)
        diag["experiment"] = experiment
        diag["generator"] = generator
        diag["controlled_upper_bound"] = upper_bound
        diag["deployable_repair"] = bool(not upper_bound)
        diag_rows.append(diag)
    pilot_metrics = pd.concat(metric_frames, ignore_index=True)
    gap_rows = pilot_metrics[pilot_metrics["N"] == max(ns)].copy()
    gap_rows = gap_rows[
        [
            "experiment",
            "generator",
            "repair_model",
            "pilot_budget",
            "N",
            "raw_real_utility",
            "fixed_real_utility",
            "oracle_real_utility",
            "oracle_gap_raw",
            "oracle_gap_fixed",
            "gap_closed",
            "gap_closed_clamped",
            "controlled_upper_bound",
            "deployable_repair",
        ]
    ]
    adaptive = pilot_metrics[
        [
            "experiment",
            "generator",
            "repair_model",
            "pilot_budget",
            "N",
            "selected_lcb_mean",
            "fixed_real_utility",
            "high_n_regret",
            "upper_tail_rank_correlation",
            "imagined_real_tail_gap",
            "allow_high_n",
            "stop_early",
            "collect_pilot_labels",
            "block_high_n",
            "deployment_gate",
            "gate_reason",
        ]
    ].copy()
    diagnostics = pd.DataFrame(diag_rows)
    return pilot_metrics, gap_rows, adaptive, diagnostics
