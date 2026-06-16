from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "results" / "v4_frozen_evidence"
FIG_OUT = OUT / "figures"
PAPER_FIG_OUT = ROOT / "figures" / "v4"
MACROS = ROOT / "paper_iclr" / "v4_results_macros.tex"
PDF_METADATA = {
    "Creator": "experiments/v4_frozen_evidence.py",
    "CreationDate": None,
    "ModDate": None,
}
N_VALUES = (1, 2, 4, 8, 16, 32, 64)
BENCHMARK_SPECS = (
    {"env_id": "CartPole-v1", "horizon": 80, "n_pools": 18, "train_pools": 8, "cal_pools": 4},
    {"env_id": "Pendulum-v1", "horizon": 64, "n_pools": 18, "train_pools": 8, "cal_pools": 4},
    {"env_id": "MountainCarContinuous-v0", "horizon": 72, "n_pools": 18, "train_pools": 8, "cal_pools": 4},
)
N_CANDIDATES = 64


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


def clean_stale_version_outputs() -> None:
    for directory in [OUT, FIG_OUT, PAPER_FIG_OUT]:
        if not directory.exists():
            continue
        for path in directory.glob("v3_*"):
            if path.is_file():
                path.unlink()


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


def _ci95(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    rx = pd.Series(x).rank(method="average").to_numpy(float)
    ry = pd.Series(y).rank(method="average").to_numpy(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _finite_law_expected(scores: np.ndarray, utilities: np.ndarray, n: int) -> float:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    utilities = np.asarray(utilities, dtype=float).reshape(-1)
    expected = 0.0
    p_higher = 0.0
    total = float(scores.size)
    for score in sorted(np.unique(scores), reverse=True):
        mask = scores == score
        p_group = float(mask.sum() / total)
        prob = (1.0 - p_higher) ** int(n) - (1.0 - p_higher - p_group) ** int(n)
        expected += prob * float(utilities[mask].mean())
        p_higher += p_group
    return float(expected)


def _monte_carlo_top_tail(scores: np.ndarray, utilities: np.ndarray, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    utilities = np.asarray(utilities, dtype=float).reshape(-1)
    selected = []
    for _ in range(900):
        idx = rng.integers(0, scores.size, size=int(n))
        local_scores = scores[idx]
        top = local_scores.max()
        tied = idx[local_scores == top]
        selected.append(float(utilities[rng.choice(tied)]))
    return float(np.mean(selected))


def _normalise_actions(actions: np.ndarray, discrete: bool) -> np.ndarray:
    arr = np.asarray(actions, dtype=float)
    if discrete:
        arr = 2.0 * arr.reshape(-1, 1) - 1.0
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(float)


def _candidate_features(initial_obs: np.ndarray, actions: np.ndarray, discrete: bool) -> np.ndarray:
    obs = np.asarray(initial_obs, dtype=float).reshape(-1)
    act = _normalise_actions(actions, discrete)
    flat_head = act[: min(12, len(act))].reshape(-1)
    if flat_head.size < 12:
        flat_head = np.pad(flat_head, (0, 12 - flat_head.size))
    flat_tail = act[-min(12, len(act)) :].reshape(-1)
    if flat_tail.size < 12:
        flat_tail = np.pad(flat_tail, (0, 12 - flat_tail.size))
    diffs = np.diff(act, axis=0) if len(act) > 1 else np.zeros_like(act)
    stats = np.array(
        [
            act.mean(),
            act.std(),
            act.min(),
            act.max(),
            np.mean(np.abs(act)),
            np.mean(act**2),
            np.mean(np.abs(diffs)),
            np.max(np.abs(diffs)) if diffs.size else 0.0,
        ],
        dtype=float,
    )
    return np.concatenate([obs, stats, flat_head, flat_tail]).astype(float)


def _sample_action_sequence(env: Any, spec: dict[str, Any], rng: np.random.Generator, candidate_id: int) -> np.ndarray:
    horizon = int(spec["horizon"])
    if hasattr(env.action_space, "n"):
        if candidate_id == 0:
            return np.zeros(horizon, dtype=int)
        if candidate_id == 1:
            return np.ones(horizon, dtype=int)
        if candidate_id == 2:
            return (np.arange(horizon) % 2).astype(int)
        persistence = float(rng.uniform(0.62, 0.94))
        value = int(rng.integers(0, env.action_space.n))
        seq = np.empty(horizon, dtype=int)
        bias = float(rng.beta(1.6, 1.6))
        for t in range(horizon):
            if t == 0 or rng.random() > persistence:
                value = int(rng.random() < bias)
            seq[t] = value
        return seq
    low = np.asarray(env.action_space.low, dtype=float).reshape(1, -1)
    high = np.asarray(env.action_space.high, dtype=float).reshape(1, -1)
    dim = int(low.size)
    if candidate_id < 3:
        constants = [low.reshape(-1), np.zeros(dim), high.reshape(-1)]
        return np.repeat(constants[candidate_id].reshape(1, -1), horizon, axis=0)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    base = rng.uniform(low, high, size=(1, dim)) * rng.uniform(0.15, 0.85)
    noise = rng.normal(0.0, 0.45, size=(horizon, dim))
    for t in range(1, horizon):
        noise[t] = 0.82 * noise[t - 1] + 0.18 * noise[t]
    wave = 0.35 * np.sin(np.linspace(0.0, 2.0 * np.pi, horizon).reshape(-1, 1) + phase)
    return np.clip(base + noise + wave, low, high).astype(float)


def _reset_to_state(env: Any, seed: int, state: np.ndarray) -> np.ndarray:
    obs, _ = env.reset(seed=int(seed))
    if hasattr(env.unwrapped, "state"):
        env.unwrapped.state = np.asarray(state, dtype=float).copy()
        obs = np.asarray(env.unwrapped.state, dtype=float).copy()
    return np.asarray(obs, dtype=float).copy()


def _rollout_candidate(env: Any, spec: dict[str, Any], initial_state: np.ndarray, actions: np.ndarray, seed: int) -> dict[str, Any]:
    obs = _reset_to_state(env, seed, initial_state)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    for action in actions:
        act = int(action) if hasattr(env.action_space, "n") else np.asarray(action, dtype=float)
        obs, reward, terminated, truncated, _ = env.step(act)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    horizon = int(spec["horizon"])
    if spec["env_id"] == "CartPole-v1":
        utility = total_reward / horizon
    elif spec["env_id"] == "Pendulum-v1":
        utility = total_reward / (8.0 * horizon)
    else:
        utility = total_reward / max(1.0, float(horizon))
    return {
        "utility": float(utility),
        "total_reward": float(total_reward),
        "steps": int(steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_obs": np.asarray(obs, dtype=float).reshape(-1),
    }


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, alpha: float) -> np.ndarray:
    train_x = np.asarray(train_x, dtype=float)
    query_x = np.asarray(query_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x = (train_x - mean) / std
    q = (query_x - mean) / std
    x_aug = np.c_[np.ones(len(x)), x]
    q_aug = np.c_[np.ones(len(q)), q]
    reg = np.eye(x_aug.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ train_y)
    return q_aug @ weights


def _ridge_ensemble(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    preds = []
    n = int(len(train_y))
    for member in range(9):
        idx = rng.integers(0, n, size=max(16, int(0.82 * n)))
        alpha = [0.05, 0.25, 1.0, 4.0, 12.0][member % 5]
        preds.append(_ridge_predict(train_x[idx], train_y[idx], query_x, alpha))
    pred = np.vstack(preds)
    return pred.mean(axis=0), pred.std(axis=0)


def _benchmark_records_for_spec(spec: dict[str, Any]) -> pd.DataFrame:
    import gymnasium as gym

    rows = []
    env = gym.make(str(spec["env_id"]))
    try:
        for pool_id in range(int(spec["n_pools"])):
            pool_seed = 2407 + 97 * pool_id + 11 * len(str(spec["env_id"]))
            obs, _ = env.reset(seed=pool_seed)
            initial_state = np.asarray(getattr(env.unwrapped, "state", obs), dtype=float).copy()
            rng = np.random.default_rng(pool_seed + 10_003)
            for candidate_id in range(N_CANDIDATES):
                actions = _sample_action_sequence(env, spec, rng, candidate_id)
                result = _rollout_candidate(env, spec, initial_state, actions, pool_seed + candidate_id)
                feature = _candidate_features(initial_state, actions, hasattr(env.action_space, "n"))
                rows.append(
                    {
                        "env_id": spec["env_id"],
                        "pool_id": pool_id,
                        "candidate_id": candidate_id,
                        "split": (
                            "train"
                            if pool_id < int(spec["train_pools"])
                            else "calibration"
                            if pool_id < int(spec["train_pools"]) + int(spec["cal_pools"])
                            else "eval"
                        ),
                        "utility": result["utility"],
                        "total_reward": result["total_reward"],
                        "steps": result["steps"],
                        "terminated": result["terminated"],
                        "truncated": result["truncated"],
                        "feature": feature,
                        "action_energy": float(np.mean(_normalise_actions(actions, hasattr(env.action_space, "n")) ** 2)),
                    }
                )
    finally:
        env.close()
    return pd.DataFrame(rows)


def build_gymnasium_benchmark_evidence() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_records = pd.concat([_benchmark_records_for_spec(spec) for spec in BENCHMARK_SPECS], ignore_index=True)
    scored_frames = []
    for env_id, group in raw_records.groupby("env_id", sort=False):
        train = group[group["split"] == "train"].copy()
        cal = group[group["split"] == "calibration"].copy()
        eval_rows = group[group["split"] == "eval"].copy()
        train_x = np.vstack(train["feature"].to_numpy())
        train_y = train["utility"].to_numpy(float)
        cal_x = np.vstack(cal["feature"].to_numpy())
        eval_x = np.vstack(eval_rows["feature"].to_numpy())
        cal_pred, cal_unc = _ridge_ensemble(train_x, train_y, cal_x, seed=910 + len(env_id))
        eval_pred, eval_unc = _ridge_ensemble(train_x, train_y, eval_x, seed=1201 + len(env_id))
        cal_resid = np.abs(cal_pred - cal["utility"].to_numpy(float))
        q_resid = float(np.quantile(cal_resid, 0.85)) if len(cal_resid) else 0.0
        eval_rows["raw_score"] = eval_pred
        eval_rows["uncertainty"] = eval_unc + q_resid
        eval_rows["lcb_score"] = eval_pred - 1.15 * eval_rows["uncertainty"].to_numpy(float) - 0.05 * eval_rows["action_energy"].to_numpy(float)
        eval_rows["anti_score"] = -eval_pred
        eval_rows["random_score"] = np.random.default_rng(3001 + len(env_id)).normal(size=len(eval_rows))
        eval_rows["oracle_score"] = eval_rows["utility"].to_numpy(float)
        eval_rows["calibration_abs_residual_q85"] = q_resid
        scored_frames.append(eval_rows)
    scored = pd.concat(scored_frames, ignore_index=True)

    curve_rows = []
    law_rows = []
    scorers = {
        "random": "random_score",
        "raw_ridge": "raw_score",
        "lcb_ridge": "lcb_score",
        "anti_ridge": "anti_score",
        "oracle": "oracle_score",
    }
    for (env_id, pool_id), pool in scored.groupby(["env_id", "pool_id"], sort=False):
        pool = pool.sort_values("candidate_id")
        utility = pool["utility"].to_numpy(float)
        random_by_n: dict[int, float] = {}
        selected_by_scorer: dict[tuple[str, int], float] = {}
        for scorer_name, score_col in scorers.items():
            scores = pool[score_col].to_numpy(float)
            for n in N_VALUES:
                local_scores = scores[: int(n)]
                idx = int(np.flatnonzero(local_scores == local_scores.max())[0])
                selected = float(utility[idx])
                selected_by_scorer[(scorer_name, int(n))] = selected
                if scorer_name == "random":
                    random_by_n[int(n)] = selected
                curve_rows.append(
                    {
                        "env_id": env_id,
                        "pool_id": int(pool_id),
                        "scorer": scorer_name,
                        "N": int(n),
                        "selected_real_utility": selected,
                        "selected_score": float(scores[idx]),
                        "oracle_real_utility": float(np.max(utility[: int(n)])),
                        "tail_rank_correlation": _rank_corr(scores, utility),
                    }
                )
                if int(n) in (8, 32, 64) and scorer_name in {"raw_ridge", "lcb_ridge", "anti_ridge"}:
                    law_expected = _finite_law_expected(scores, utility, int(n))
                    mc_expected = _monte_carlo_top_tail(scores, utility, int(n), seed=5000 + int(pool_id) + int(n))
                    law_rows.append(
                        {
                            "env_id": env_id,
                            "pool_id": int(pool_id),
                            "scorer": scorer_name,
                            "N": int(n),
                            "law_expected_utility": law_expected,
                            "mc_expected_utility": mc_expected,
                            "abs_error": abs(law_expected - mc_expected),
                        }
                    )
        for scorer_name in ["raw_ridge", "lcb_ridge", "anti_ridge", "oracle"]:
            for n in N_VALUES:
                selected_key = (scorer_name, int(n))
                curve_rows.append(
                    {
                        "env_id": env_id,
                        "pool_id": int(pool_id),
                        "scorer": f"{scorer_name}_minus_random",
                        "N": int(n),
                        "selected_real_utility": selected_by_scorer[selected_key] - random_by_n[int(n)],
                        "selected_score": np.nan,
                        "oracle_real_utility": np.nan,
                        "tail_rank_correlation": np.nan,
                    }
                )
    curve = pd.DataFrame(curve_rows)
    law = pd.DataFrame(law_rows)

    summary_rows = []
    for (env_id, scorer), rows in curve[curve["N"] == 64].groupby(["env_id", "scorer"], sort=False):
        vals = rows["selected_real_utility"].to_numpy(float)
        if scorer.endswith("_minus_random"):
            ci_lo = float(vals.mean() - _ci95(vals))
            ci_hi = float(vals.mean() + _ci95(vals))
        else:
            ci_lo = float(vals.mean() - _ci95(vals))
            ci_hi = float(vals.mean() + _ci95(vals))
        summary_rows.append(
            {
                "env_id": env_id,
                "scorer": scorer,
                "eval_pools": int(rows["pool_id"].nunique()),
                "selected_real_mean_n64": float(vals.mean()),
                "selected_real_ci_lo_n64": ci_lo,
                "selected_real_ci_hi_n64": ci_hi,
            }
        )
    summary = pd.DataFrame(summary_rows)
    return scored.drop(columns=["feature"]), curve, law, summary


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
    clean_stale_version_outputs()

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
    n64.to_csv(OUT / "v4_n64_tail_failures.csv", index=False)

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
    seed_robustness.to_csv(OUT / "v4_seed_robustness.csv", index=False)

    gap_sorted = gap.sort_values(["experiment", "repair_model", "pilot_budget"]).copy()
    gap_sorted.to_csv(OUT / "v4_repair_budget.csv", index=False)

    denoising_out = denoising.sort_values(["denoising_steps", "N"]).copy()
    denoising_out.to_csv(OUT / "v4_denoising_grid.csv", index=False)

    calibration_sorted = calibration.sort_values(["experiment", "repair_model", "pilot_budget"]).copy()
    calibration_sorted.to_csv(OUT / "v4_calibration_diagnostics.csv", index=False)

    benchmark_candidates, benchmark_curve, benchmark_law, benchmark_summary = build_gymnasium_benchmark_evidence()
    benchmark_candidates.to_csv(OUT / "v4_benchmark_candidates.csv", index=False)
    benchmark_curve.to_csv(OUT / "v4_benchmark_selection_curves.csv", index=False)
    benchmark_law.to_csv(OUT / "v4_benchmark_law_validation.csv", index=False)
    benchmark_summary.to_csv(OUT / "v4_benchmark_summary.csv", index=False)

    from dwm_tail_audit.audit import write_claim_audit

    write_claim_audit(ROOT)
    claims_payload = read_json("results/claims_status.json")
    claim_inventory = write_claim_inventory(claims_payload)
    claim_inventory.to_csv(OUT / "v4_claim_inventory.csv", index=False)

    inventory = artifact_inventory()
    inventory_summary = (
        inventory.groupby(["root", "suffix"], as_index=False)
        .agg(files=("path", "count"), bytes=("bytes", "sum"))
        .sort_values(["root", "suffix"])
    )
    inventory_summary.to_csv(OUT / "v4_artifact_inventory.csv", index=False)

    # Figure 1: N=64 selected-tail failure landscape.
    failure_plot = n64[n64["scorer"].isin(["raw", "calibrated"])].copy()
    failure_plot["label"] = failure_plot["generator"] + " / " + failure_plot["scorer"]
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    ax.barh(failure_plot["label"], failure_plot["imagined_real_tail_gap"], color="#9b2c2c")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("imagined-real tail gap at N=64")
    ax.set_title("Selected-tail hallucination severity")
    savefig(fig, FIG_OUT / "v4_tail_failure_landscape.pdf")

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
    savefig(fig, FIG_OUT / "v4_seed_robustness.pdf")

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
    savefig(fig, FIG_OUT / "v4_repair_budget_curve.pdf")

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
    savefig(fig, FIG_OUT / "v4_denoising_selection_heatmap.pdf")

    # Figure 5: calibration coverage.
    cal_plot = calibration_sorted[calibration_sorted["pilot_budget"] > 0].copy()
    cal_plot["label"] = cal_plot["experiment"] + " / " + cal_plot["repair_model"] + " / " + cal_plot["pilot_budget"].astype(str)
    fig, ax = plt.subplots(figsize=(8.2, max(3.8, 0.35 * len(cal_plot))))
    ax.barh(cal_plot["label"], cal_plot["eval_lower_bound_coverage"].fillna(0.0), color="#238b45")
    ax.axvline(0.9, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("held-out lower-bound coverage")
    ax.set_title("Calibration coverage by repair setting")
    savefig(fig, FIG_OUT / "v4_calibration_coverage.pdf")

    # Figure 6: claim and artifact inventory.
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.8))
    axes[0].bar(claim_inventory["status"], claim_inventory["count"], color=["#225ea8", "#fd8d3c", "#bdbdbd"])
    axes[0].set_title("Claim ledger")
    axes[0].set_ylabel("claims")
    suffix_counts = inventory.groupby("suffix")["path"].count().sort_values(ascending=False).head(6)
    axes[1].bar(suffix_counts.index, suffix_counts.values, color="#756bb1")
    axes[1].set_title("Artifact suffixes")
    axes[1].set_ylabel("files")
    savefig(fig, FIG_OUT / "v4_claim_artifact_inventory.pdf")

    # Figure 7: standard Gymnasium benchmark selection baselines.
    bench_plot = benchmark_summary[
        benchmark_summary["scorer"].isin(["random", "raw_ridge", "lcb_ridge", "oracle"])
    ].copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    labels = bench_plot["env_id"] + " / " + bench_plot["scorer"]
    ax.barh(labels, bench_plot["selected_real_mean_n64"], color="#2b8cbe")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("selected real utility at N=64")
    ax.set_title("Gymnasium benchmark rollout-pool baselines")
    savefig(fig, FIG_OUT / "v4_gymnasium_benchmark_baselines.pdf")

    # Figure 8: benchmark stress deltas against random selection.
    delta_plot = benchmark_summary[benchmark_summary["scorer"].str.endswith("_minus_random")].copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    labels = delta_plot["env_id"] + " / " + delta_plot["scorer"].str.replace("_minus_random", "", regex=False)
    means = delta_plot["selected_real_mean_n64"].to_numpy(float)
    lo = means - delta_plot["selected_real_ci_lo_n64"].to_numpy(float)
    hi = delta_plot["selected_real_ci_hi_n64"].to_numpy(float) - means
    y = np.arange(len(delta_plot))
    ax.errorbar(means, y, xerr=[lo, hi], fmt="o", color="#54278f", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("selected real utility minus random at N=64")
    ax.set_title("Benchmark baselines and negative controls")
    savefig(fig, FIG_OUT / "v4_gymnasium_benchmark_deltas.pdf")

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
    benchmark_delta = benchmark_summary[benchmark_summary["scorer"].str.endswith("_minus_random")].copy()
    positive_ci_rows = int(
        (
            benchmark_delta["scorer"].isin(["raw_ridge_minus_random", "lcb_ridge_minus_random"])
            & (benchmark_delta["selected_real_ci_lo_n64"] > 0.0)
        ).sum()
    )
    anti_negative_rows = int(
        (
            (benchmark_delta["scorer"] == "anti_ridge_minus_random")
            & (benchmark_delta["selected_real_ci_hi_n64"] < 0.0)
        ).sum()
    )
    lcb_delta = benchmark_delta[benchmark_delta["scorer"] == "lcb_ridge_minus_random"]
    benchmark_eval_pools = int(
        benchmark_candidates[benchmark_candidates["split"] == "eval"][["env_id", "pool_id"]].drop_duplicates().shape[0]
    )

    summary = {
        "supported_claims": supported,
        "partial_claims": partial,
        "unsupported_boundary_claims": unsupported,
        "artifact_files": int(len(inventory)),
        "result_table_files": int(len(list(TABLES.glob("*.csv")))),
        "v4_figure_files": int(len(list(FIG_OUT.glob("*.pdf")))),
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
        "benchmark_envs": int(benchmark_candidates["env_id"].nunique()),
        "benchmark_eval_pools": benchmark_eval_pools,
        "benchmark_candidate_rows": int(len(benchmark_candidates)),
        "benchmark_curve_rows": int(len(benchmark_curve)),
        "benchmark_positive_ci_rows": positive_ci_rows,
        "benchmark_anti_negative_rows": anti_negative_rows,
        "benchmark_lcb_min_ci_lo": float(lcb_delta["selected_real_ci_lo_n64"].min()) if len(lcb_delta) else 0.0,
        "benchmark_law_max_abs_error": float(benchmark_law["abs_error"].max()),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with MACROS.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by experiments/v4_frozen_evidence.py\n")
        handle.write(macro_line("VFourDWMSupportedClaims", supported))
        handle.write(macro_line("VFourDWMPartialClaims", partial))
        handle.write(macro_line("VFourDWMUnsupportedBoundaryClaims", unsupported))
        handle.write(macro_line("VFourDWMArtifactFiles", summary["artifact_files"]))
        handle.write(macro_line("VFourDWMResultTables", summary["result_table_files"]))
        handle.write(macro_line("VFourDWMSeedRows", summary["seed_rows"]))
        handle.write(macro_line("VFourDWMDenoisingRows", summary["denoising_rows"]))
        handle.write(macro_line("VFourDWMPilotRows", summary["pilot_repair_rows"]))
        handle.write(macro_line("VFourDWMCalibrationRows", summary["calibration_rows"]))
        handle.write(macro_line("VFourDWMRunSeconds", summary["run_elapsed_seconds"], 1))
        handle.write(macro_line("VFourDWMRunSeeds", summary["run_seeds"]))
        handle.write(macro_line("VFourDWMRunConditions", summary["run_conditions"]))
        handle.write(macro_line("VFourDWMLawTrials", summary["law_trials"]))
        handle.write(macro_line("VFourDWMExactLawMaxError", exact_max, 4))
        handle.write(macro_line("VFourDWMGoodRawReal", summary["good_raw_n64_real"], 3))
        handle.write(macro_line("VFourDWMOptimisticImagined", summary["optimistic_raw_n64_imagined"], 3))
        handle.write(macro_line("VFourDWMOptimisticReal", summary["optimistic_raw_n64_real"], 3))
        handle.write(macro_line("VFourDWMOptimisticTailGap", summary["optimistic_raw_n64_tail_gap"], 3))
        handle.write(macro_line("VFourDWMOptimisticRegret", summary["optimistic_raw_n64_high_regret"], 3))
        handle.write(macro_line("VFourDWMModeCollapsedTailGap", summary["mode_collapsed_n64_tail_gap"], 3))
        handle.write(macro_line("VFourDWMPlausibilityTailGap", summary["plausibility_biased_n64_tail_gap"], 3))
        handle.write(macro_line("VFourDWMLearnedRawImagined", summary["learned_raw_n64_imagined"], 3))
        handle.write(macro_line("VFourDWMLearnedRawReal", summary["learned_raw_n64_real"], 3))
        handle.write(macro_line("VFourDWMLearnedRawTailGap", summary["learned_raw_n64_tail_gap"], 3))
        handle.write(macro_line("VFourDWMLearnedRawRegret", summary["learned_raw_n64_high_regret"], 3))
        handle.write(macro_line("VFourDWMLearnedCalReal", summary["learned_cal_n64_real"], 3))
        handle.write(macro_line("VFourDWMControlledGapBThirtyTwo", summary["controlled_gap_closed_budget32"], 3))
        handle.write(macro_line("VFourDWMControlledGapBOneTwentyEight", summary["controlled_gap_closed_budget128"], 3))
        handle.write(macro_line("VFourDWMLearnedGapBThirtyTwo", summary["learned_gap_closed_budget32"], 3))
        handle.write(macro_line("VFourDWMOracleFeatureGap", summary["oracle_feature_gap_closed"], 3))
        handle.write(macro_line("VFourDWMLearnedTestMSE", summary["learned_test_mse"], 3))
        handle.write(macro_line("VFourDWMLearnedFinalError", summary["learned_test_final_state_error"], 3))
        handle.write(macro_line("VFourDWMLearnedHeldoutRankCorr", summary["learned_heldout_rank_correlation"], 3))
        handle.write(macro_line("VFourDWMLearnedHeldoutDiversity", summary["learned_heldout_sample_diversity"], 3))
        handle.write(macro_line("VFourDWMLearnedTailCalibrationError", summary["learned_tail_calibration_error"], 3))
        handle.write(macro_line("VFourDWMTrainingLossFirst", summary["training_loss_first"], 3))
        handle.write(macro_line("VFourDWMTrainingLossLast", summary["training_loss_last"], 3))
        handle.write(macro_line("VFourDWMBenchmarkEnvs", summary["benchmark_envs"]))
        handle.write(macro_line("VFourDWMBenchmarkEvalPools", summary["benchmark_eval_pools"]))
        handle.write(macro_line("VFourDWMBenchmarkCandidateRows", summary["benchmark_candidate_rows"]))
        handle.write(macro_line("VFourDWMBenchmarkCurveRows", summary["benchmark_curve_rows"]))
        handle.write(macro_line("VFourDWMBenchmarkPositiveCIRows", summary["benchmark_positive_ci_rows"]))
        handle.write(macro_line("VFourDWMBenchmarkAntiNegativeRows", summary["benchmark_anti_negative_rows"]))
        handle.write(macro_line("VFourDWMBenchmarkLCBMinCILo", summary["benchmark_lcb_min_ci_lo"], 4))
        handle.write(macro_line("VFourDWMBenchmarkLawMaxError", summary["benchmark_law_max_abs_error"], 4))

    return summary


def main() -> None:
    summary = build_outputs()
    print(f"v4 cached evidence complete: {OUT}")
    print(
        "claims={supported} tables={tables} seed_rows={seed_rows} figures={figures}".format(
            supported=summary["supported_claims"],
            tables=summary["result_table_files"],
            seed_rows=summary["seed_rows"],
            figures=summary["v4_figure_files"],
        )
    )


if __name__ == "__main__":
    main()
