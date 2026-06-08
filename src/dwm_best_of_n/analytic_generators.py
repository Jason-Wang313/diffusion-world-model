"""Controlled diffusion-like future generators for selected-tail experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .toy_world import Condition, ToyWorld


GENERATOR_VARIANTS = ("good", "optimistic", "mode_collapsed", "plausibility_biased")


@dataclass
class CandidateBatch:
    generator: str
    condition_id: int
    state: np.ndarray
    goal: np.ndarray
    action_sequences: np.ndarray
    future_states: np.ndarray
    real_utility: np.ndarray
    imagined_score: np.ndarray
    plausibility: np.ndarray
    uncertainty: np.ndarray
    consistency: np.ndarray
    modes: np.ndarray
    metadata: dict[str, float | str] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.real_utility.shape[0])


def imagined_utility_from_future(future_states: np.ndarray, actions: np.ndarray, goal: np.ndarray) -> np.ndarray:
    final_distance = np.linalg.norm(future_states[:, -1, :] - goal.reshape(1, 2), axis=1)
    effort = np.mean(np.linalg.norm(actions, axis=2) ** 2, axis=1)
    smooth = np.mean(np.linalg.norm(np.diff(actions, axis=1), axis=2), axis=1)
    return 1.35 - 1.18 * final_distance - 0.016 * effort - 0.06 * smooth


def _diversity_noise(rng: np.random.Generator, shape: tuple[int, ...], denoising_steps: int) -> np.ndarray:
    sigma = 0.18 / np.sqrt(max(denoising_steps, 1))
    return rng.normal(0.0, sigma, size=shape).astype(np.float32)


def generate_analytic_batch(
    variant: str,
    world: ToyWorld,
    condition: Condition,
    n: int,
    seed: int,
    denoising_steps: int = 8,
) -> CandidateBatch:
    if variant not in GENERATOR_VARIANTS:
        raise ValueError(f"unknown generator variant {variant!r}")
    rng = np.random.default_rng(seed)
    actions = world.sample_candidate_actions(condition.state, condition.goal, n, rng)

    real_states = np.empty((n, world.config.horizon, 2), dtype=np.float32)
    free_states = np.empty_like(real_states)
    real_utility = np.empty(n, dtype=np.float32)
    free_utility = np.empty(n, dtype=np.float32)
    modes = np.empty(n, dtype=object)
    for i in range(n):
        real_rollout = world.rollout(
            condition.state,
            actions[i],
            condition.goal,
            condition.mode,
            rng=np.random.default_rng(seed * 7919 + i),
        )
        free_rollout = world.free_mode_rollout(condition.state, actions[i], condition.goal)
        real_states[i] = real_rollout.states
        free_states[i] = free_rollout.states
        real_utility[i] = real_rollout.utility
        free_utility[i] = free_rollout.utility
        modes[i] = condition.mode

    speed = np.mean(np.linalg.norm(actions, axis=2), axis=1)
    smooth = np.mean(np.linalg.norm(np.diff(actions, axis=1), axis=2), axis=1)
    approx_path = condition.state.reshape(1, 1, 2) + np.cumsum(actions * world.config.dt, axis=1)
    barrier_hazard = ((approx_path[:, :, 0] > 0.08) & (np.abs(approx_path[:, :, 1]) < 0.34)).any(axis=1).astype(float)
    fragile_hazard = (np.max(np.linalg.norm(actions, axis=2), axis=1) > 0.92).astype(float)
    risk = np.clip((speed - 0.72) / 0.75, 0.0, 1.6)
    physical_risk = np.clip(risk + 0.85 * barrier_hazard + 0.45 * fragile_hazard + 0.25 * smooth, 0.0, 2.2)
    denoise_factor = 1.0 / np.sqrt(max(denoising_steps, 1))

    if variant == "good":
        future = real_states + _diversity_noise(rng, real_states.shape, denoising_steps)
        imagined = real_utility + rng.normal(0.0, 0.05 + 0.05 * denoise_factor, size=n)
        uncertainty = 0.08 + 0.12 * denoise_factor + 0.05 * physical_risk
        consistency = np.clip(0.90 - 0.08 * physical_risk + rng.normal(0.0, 0.025, size=n), 0.0, 1.0)
    elif variant == "optimistic":
        # The model dreams the free/smooth outcome and gives the aggressive tail
        # extra reward, especially at low denoising budgets.
        future = free_states + _diversity_noise(rng, free_states.shape, denoising_steps)
        toward_goal = condition.goal.reshape(1, 1, 2) - future
        future = future + (0.08 + 0.20 * risk[:, None, None]) * toward_goal
        imagined = imagined_utility_from_future(future, actions, condition.goal)
        imagined += 0.35 * risk + 0.28 * denoise_factor * risk
        imagined += rng.normal(0.0, 0.08 + 0.09 * denoise_factor, size=n)
        uncertainty = 0.18 + 0.22 * denoise_factor + 0.32 * risk + 0.50 * barrier_hazard + 0.24 * fragile_hazard
        consistency = np.clip(
            0.82 - 0.20 * risk - 0.46 * barrier_hazard - 0.18 * fragile_hazard + rng.normal(0.0, 0.05, size=n),
            0.0,
            1.0,
        )
    elif variant == "mode_collapsed":
        future = free_states + _diversity_noise(rng, free_states.shape, denoising_steps)
        imagined = imagined_utility_from_future(future, actions, condition.goal)
        hidden_mismatch = 0.35 if condition.mode in {"blocked", "slip", "fragile"} else 0.05
        imagined += hidden_mismatch + 0.16 * risk + rng.normal(0.0, 0.07, size=n)
        uncertainty = 0.14 + 0.14 * denoise_factor + 0.16 * risk + 0.42 * barrier_hazard
        consistency = np.clip(0.84 - 0.17 * risk - 0.38 * barrier_hazard - 0.10 * (condition.mode == "blocked"), 0.0, 1.0)
    else:
        # Plausible-looking smooth futures can be wrong under hidden blockage.
        future = 0.72 * free_states + 0.28 * real_states + _diversity_noise(rng, real_states.shape, denoising_steps)
        plausibility_pref = 0.65 * np.exp(-0.55 * smooth) + 0.28 * np.exp(-0.35 * speed)
        imagined = imagined_utility_from_future(future, actions, condition.goal)
        imagined += 0.55 * plausibility_pref + 0.08 * rng.normal(size=n)
        uncertainty = 0.12 + 0.16 * denoise_factor + 0.12 * risk + 0.34 * barrier_hazard
        consistency = np.clip(0.86 - 0.12 * risk - 0.32 * barrier_hazard - 0.18 * (condition.mode == "blocked"), 0.0, 1.0)

    plausibility = np.clip(1.0 - 0.24 * speed - 0.38 * smooth - 0.08 * barrier_hazard + rng.normal(0.0, 0.035, size=n), 0.0, 1.0)
    if variant == "plausibility_biased":
        plausibility = np.clip(plausibility + 0.20, 0.0, 1.0)

    return CandidateBatch(
        generator=variant,
        condition_id=condition.condition_id,
        state=condition.state,
        goal=condition.goal,
        action_sequences=actions,
        future_states=future.astype(np.float32),
        real_utility=real_utility.astype(np.float32),
        imagined_score=np.asarray(imagined, dtype=np.float32),
        plausibility=np.asarray(plausibility, dtype=np.float32),
        uncertainty=np.asarray(uncertainty, dtype=np.float32),
        consistency=np.asarray(consistency, dtype=np.float32),
        modes=modes,
        metadata={"real_mode": condition.mode, "denoising_steps": float(denoising_steps)},
    )
