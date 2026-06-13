"""Scoring rules for selecting among generated futures."""

from __future__ import annotations

import numpy as np

from .analytic_generators import CandidateBatch, imagined_utility_from_future


SCORER_NAMES = (
    "raw",
    "goal_proximity",
    "plausibility",
    "calibrated",
    "uncertainty_aware",
    "consistency_aware",
    "random",
    "oracle",
)


def _action_risk(batch: CandidateBatch) -> np.ndarray:
    actions = batch.action_sequences
    speed = np.mean(np.linalg.norm(actions, axis=2), axis=1)
    smooth = np.mean(np.linalg.norm(np.diff(actions, axis=1), axis=2), axis=1)
    approx_path = batch.state.reshape(1, 1, 2) + np.cumsum(actions * 0.25, axis=1)
    barrier = ((approx_path[:, :, 0] > 0.08) & (np.abs(approx_path[:, :, 1]) < 0.34)).any(axis=1).astype(float)
    fragile = (np.max(np.linalg.norm(actions, axis=2), axis=1) > 0.92).astype(float)
    speed_tail = np.clip((speed - 0.68) / 0.72, 0.0, None)
    return speed_tail + 0.55 * smooth + 0.95 * barrier + 0.45 * fragile


def score_candidates(
    batch: CandidateBatch,
    scorer: str = "raw",
    seed: int | None = None,
) -> np.ndarray:
    if scorer not in SCORER_NAMES:
        raise ValueError(f"unknown scorer {scorer!r}")
    rng = np.random.default_rng(seed)
    if scorer == "raw":
        return batch.imagined_score.astype(float)
    if scorer == "goal_proximity":
        return imagined_utility_from_future(batch.future_states, batch.action_sequences, batch.goal).astype(float)
    if scorer == "plausibility":
        return (0.55 * batch.imagined_score + 0.95 * batch.plausibility).astype(float)
    if scorer == "calibrated":
        risk = _action_risk(batch)
        return (
            0.62 * batch.imagined_score
            + 0.70 * batch.plausibility
            + 0.85 * batch.consistency
            - 2.10 * batch.uncertainty
            - 0.85 * risk
        ).astype(float)
    if scorer == "uncertainty_aware":
        return (batch.imagined_score - 2.75 * batch.uncertainty - 0.45 * _action_risk(batch)).astype(float)
    if scorer == "consistency_aware":
        risk = _action_risk(batch)
        return (
            batch.imagined_score
            + 0.45 * batch.plausibility
            - 2.20 * (1.0 - batch.consistency)
            - 0.55 * batch.uncertainty
            - 0.65 * risk
        ).astype(float)
    if scorer == "random":
        return rng.normal(size=len(batch)).astype(float)
    return batch.real_utility.astype(float)
