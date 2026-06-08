"""Finite tie-aware Best-of-N laws and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BestOfNLawResult:
    n: int
    expected_utility: float
    expected_score: float
    group_probabilities: tuple[float, ...]
    group_mean_utilities: tuple[float, ...]
    group_scores: tuple[float, ...]


def _as_1d(name: str, values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _score_groups(scores: np.ndarray, utilities: np.ndarray) -> list[tuple[float, float, float]]:
    unique_scores = np.array(sorted(np.unique(scores), reverse=True), dtype=float)
    groups: list[tuple[float, float, float]] = []
    total = float(scores.size)
    for score in unique_scores:
        mask = scores == score
        groups.append((float(score), float(mask.sum() / total), float(utilities[mask].mean())))
    return groups


def finite_best_of_n_law(
    scores: Iterable[float],
    utilities: Iterable[float],
    n: int,
) -> BestOfNLawResult:
    """Exact expected utility for iid finite-pool Best-of-N with top-score tie handling.

    Candidates are sampled iid uniformly from a finite empirical pool. If several sampled
    candidates share the top score, the selected candidate is uniform over that top-score
    tie set. Since all tied candidates belong to the same score group, the conditional
    expected utility is the mean utility of that group.
    """

    scores_arr = _as_1d("scores", scores)
    utilities_arr = _as_1d("utilities", utilities)
    if scores_arr.shape != utilities_arr.shape:
        raise ValueError("scores and utilities must have the same shape")
    if n < 1:
        raise ValueError("n must be >= 1")

    groups = _score_groups(scores_arr, utilities_arr)
    p_higher = 0.0
    expected_utility = 0.0
    expected_score = 0.0
    group_probs: list[float] = []
    group_utils: list[float] = []
    group_scores: list[float] = []
    for score, p_group, mean_utility in groups:
        # Highest observed group is this group when no higher group appears and
        # at least one member of this group appears.
        prob_highest_here = (1.0 - p_higher) ** n - (1.0 - p_higher - p_group) ** n
        expected_utility += prob_highest_here * mean_utility
        expected_score += prob_highest_here * score
        p_higher += p_group
        group_probs.append(float(prob_highest_here))
        group_utils.append(mean_utility)
        group_scores.append(score)

    return BestOfNLawResult(
        n=int(n),
        expected_utility=float(expected_utility),
        expected_score=float(expected_score),
        group_probabilities=tuple(group_probs),
        group_mean_utilities=tuple(group_utils),
        group_scores=tuple(group_scores),
    )


def finite_best_of_n_curve(
    scores: Iterable[float],
    utilities: Iterable[float],
    ns: Iterable[int],
) -> pd.DataFrame:
    rows = []
    for n in ns:
        law = finite_best_of_n_law(scores, utilities, int(n))
        rows.append(
            {
                "N": int(n),
                "expected_utility": law.expected_utility,
                "expected_score": law.expected_score,
            }
        )
    return pd.DataFrame(rows)


def binary_best_of_n_law(scores: Iterable[float], utilities: Iterable[int], n: int) -> BestOfNLawResult:
    utilities_arr = _as_1d("utilities", utilities)
    if not np.all((utilities_arr == 0.0) | (utilities_arr == 1.0)):
        raise ValueError("binary utilities must be 0/1")
    return finite_best_of_n_law(scores, utilities_arr, n)


def real_valued_best_of_n_law(scores: Iterable[float], utilities: Iterable[float], n: int) -> BestOfNLawResult:
    return finite_best_of_n_law(scores, utilities, n)


def monte_carlo_best_of_n(
    scores: Iterable[float],
    utilities: Iterable[float],
    n: int,
    trials: int = 20_000,
    seed: int = 0,
) -> dict[str, float]:
    scores_arr = _as_1d("scores", scores)
    utilities_arr = _as_1d("utilities", utilities)
    if scores_arr.shape != utilities_arr.shape:
        raise ValueError("scores and utilities must have the same shape")
    rng = np.random.default_rng(seed)
    selected_utilities = np.empty(trials, dtype=float)
    selected_scores = np.empty(trials, dtype=float)
    for t in range(trials):
        idx = rng.integers(0, scores_arr.size, size=n)
        sampled_scores = scores_arr[idx]
        top = sampled_scores.max()
        tied = idx[sampled_scores == top]
        chosen = rng.choice(tied)
        selected_utilities[t] = utilities_arr[chosen]
        selected_scores[t] = scores_arr[chosen]
    return {
        "mc_expected_utility": float(selected_utilities.mean()),
        "mc_expected_score": float(selected_scores.mean()),
        "mc_se_utility": float(selected_utilities.std(ddof=1) / np.sqrt(trials)),
    }


def rank_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    """Spearman rank correlation with graceful constant-vector handling."""

    x_arr = _as_1d("x", x)
    y_arr = _as_1d("y", y)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    if x_arr.size < 2 or np.all(x_arr == x_arr[0]) or np.all(y_arr == y_arr[0]):
        return 0.0
    rx = pd.Series(x_arr).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y_arr).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(rx, ry)[0, 1])


def upper_tail_rank_correlation(
    scores: Iterable[float],
    utilities: Iterable[float],
    quantile: float = 0.8,
) -> float:
    scores_arr = _as_1d("scores", scores)
    utilities_arr = _as_1d("utilities", utilities)
    if scores_arr.shape != utilities_arr.shape:
        raise ValueError("scores and utilities must have the same shape")
    cutoff = float(np.quantile(scores_arr, quantile))
    mask = scores_arr >= cutoff
    if mask.sum() < 2:
        return 0.0
    return rank_correlation(scores_arr[mask], utilities_arr[mask])


def binary_auc(scores: Iterable[float], labels: Iterable[int]) -> float:
    """Mann-Whitney AUC for binary labels; returns nan for one-class labels."""

    scores_arr = _as_1d("scores", scores)
    labels_arr = _as_1d("labels", labels)
    if scores_arr.shape != labels_arr.shape:
        raise ValueError("scores and labels must have the same shape")
    pos = labels_arr == 1
    neg = labels_arr == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = pd.Series(scores_arr).rank(method="average").to_numpy(dtype=float)
    pos_rank_sum = ranks[pos].sum()
    n_pos = float(pos.sum())
    n_neg = float(neg.sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def tail_diagnostics(
    imagined_scores: Iterable[float],
    real_utilities: Iterable[float],
    quantile: float = 0.8,
) -> dict[str, float]:
    scores = _as_1d("imagined_scores", imagined_scores)
    utilities = _as_1d("real_utilities", real_utilities)
    cutoff = float(np.quantile(scores, quantile))
    mask = scores >= cutoff
    if mask.sum() == 0:
        return {
            "upper_tail_rank_correlation": 0.0,
            "tail_mean_imagined_score": float("nan"),
            "tail_mean_real_utility": float("nan"),
            "imagined_real_tail_gap": float("nan"),
        }
    tail_score = float(scores[mask].mean())
    tail_real = float(utilities[mask].mean())
    return {
        "upper_tail_rank_correlation": upper_tail_rank_correlation(scores, utilities, quantile),
        "tail_mean_imagined_score": tail_score,
        "tail_mean_real_utility": tail_real,
        "imagined_real_tail_gap": float(tail_score - tail_real),
    }


def exact_law_validation_dataframe(
    ns: Iterable[int] = (1, 2, 4, 8, 16, 32, 64),
    trials: int = 30_000,
    seed: int = 7,
) -> pd.DataFrame:
    """Reusable validation example with ties and real-valued utilities."""

    scores = np.array([3, 3, 2, 2, 1, 0, 0], dtype=float)
    utilities = np.array([0.1, 1.0, 0.4, -0.1, 0.2, 0.0, -0.3], dtype=float)
    rows = []
    for n in ns:
        law = finite_best_of_n_law(scores, utilities, int(n))
        mc = monte_carlo_best_of_n(scores, utilities, int(n), trials=trials, seed=seed + int(n))
        rows.append(
            {
                "N": int(n),
                "law_expected_utility": law.expected_utility,
                "mc_expected_utility": mc["mc_expected_utility"],
                "abs_error": abs(law.expected_utility - mc["mc_expected_utility"]),
                "mc_se_utility": mc["mc_se_utility"],
            }
        )
    return pd.DataFrame(rows)
