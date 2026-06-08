"""A low-dimensional hidden-friction world for controlled Best-of-N tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MODE_NAMES = ("free", "slip", "blocked", "fragile")


@dataclass(frozen=True)
class ToyWorldConfig:
    horizon: int = 12
    dt: float = 0.25
    action_limit: float = 1.6
    goal_radius: float = 0.12


@dataclass(frozen=True)
class Condition:
    state: np.ndarray
    goal: np.ndarray
    mode: str
    condition_id: int


@dataclass(frozen=True)
class RolloutResult:
    states: np.ndarray
    utility: float
    mode: str
    final_distance: float
    blockage_penalty: float
    fragility_penalty: float
    slip_penalty: float
    effort: float


class ToyWorld:
    """2D point object with hidden slip, blockage, and fragility modes."""

    def __init__(self, config: ToyWorldConfig | None = None):
        self.config = config or ToyWorldConfig()

    @property
    def feature_dim(self) -> int:
        return 2 + self.config.horizon * 2 + 2

    @property
    def target_dim(self) -> int:
        return self.config.horizon * 2

    def sample_mode(self, rng: np.random.Generator) -> str:
        return str(rng.choice(MODE_NAMES, p=[0.34, 0.26, 0.24, 0.16]))

    def sample_condition(self, rng: np.random.Generator, condition_id: int = 0) -> Condition:
        state = rng.uniform([-0.95, -0.75], [-0.45, 0.75]).astype(np.float32)
        goal = rng.uniform([0.55, -0.7], [1.05, 0.7]).astype(np.float32)
        return Condition(state=state, goal=goal, mode=self.sample_mode(rng), condition_id=condition_id)

    def base_action(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        delta = np.asarray(goal, dtype=float) - np.asarray(state, dtype=float)
        per_step = delta / max(self.config.horizon * self.config.dt, 1e-6)
        norm = np.linalg.norm(per_step)
        if norm > self.config.action_limit:
            per_step = per_step / norm * self.config.action_limit
        return per_step.astype(np.float32)

    def sample_candidate_actions(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        base = self.base_action(state, goal)
        perpendicular = np.array([-base[1], base[0]], dtype=np.float32)
        perp_norm = float(np.linalg.norm(perpendicular))
        if perp_norm > 1e-6:
            perpendicular /= perp_norm
        actions = np.empty((n, self.config.horizon, 2), dtype=np.float32)
        for i in range(n):
            # More samples expose more extreme aggressive plans, which is the
            # intended selected-tail pressure in the controlled experiments.
            gain = rng.uniform(0.55, 2.15)
            wiggle = rng.normal(0.0, 0.10, size=self.config.horizon)
            ramp = np.linspace(-0.5, 0.5, self.config.horizon)
            seq = gain * base[None, :] + (wiggle + 0.08 * ramp * rng.normal())[:, None] * perpendicular[None, :]
            seq += rng.normal(0.0, 0.035, size=seq.shape)
            norm = np.linalg.norm(seq, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.config.action_limit / np.maximum(norm, 1e-6))
            actions[i] = (seq * scale).astype(np.float32)
        rng.shuffle(actions, axis=0)
        return actions

    def rollout(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        goal: np.ndarray,
        mode: str,
        rng: np.random.Generator | None = None,
    ) -> RolloutResult:
        if mode not in MODE_NAMES:
            raise ValueError(f"unknown mode {mode!r}")
        rng = rng or np.random.default_rng(0)
        pos = np.asarray(state, dtype=np.float32).copy()
        states = np.empty((self.config.horizon, 2), dtype=np.float32)
        blockage_penalty = 0.0
        fragility_penalty = 0.0
        slip_penalty = 0.0
        effort = 0.0
        for t, raw_action in enumerate(np.asarray(actions, dtype=np.float32)):
            action = np.clip(raw_action, -self.config.action_limit, self.config.action_limit)
            action_norm = float(np.linalg.norm(action))
            effort += action_norm**2
            friction = 1.0
            drift = np.zeros(2, dtype=np.float32)
            if mode == "slip":
                friction = 0.68
                drift = np.array([0.02, 0.16 * np.sin(1.7 * t)], dtype=np.float32)
                slip_penalty += 0.018 + 0.015 * action_norm
            elif mode == "blocked":
                crossing_barrier = pos[0] < 0.08 and pos[0] + self.config.dt * action[0] > 0.08
                near_gate = abs(float(pos[1])) < 0.28
                if crossing_barrier and near_gate:
                    friction = 0.12
                    blockage_penalty += 0.55 + 0.12 * action_norm
                elif pos[0] > 0.05 and near_gate:
                    friction = 0.45
                    blockage_penalty += 0.04
            elif mode == "fragile" and action_norm > 0.92:
                friction = 0.52
                fragility_penalty += 0.14 * (action_norm - 0.92) ** 2 + 0.045
            pos = pos + self.config.dt * friction * action + drift
            if mode == "slip":
                pos += rng.normal(0.0, 0.01, size=2).astype(np.float32)
            states[t] = pos

        final_distance = float(np.linalg.norm(pos - np.asarray(goal, dtype=np.float32)))
        smoothness = float(np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=1))) if len(actions) > 1 else 0.0
        utility = (
            1.35
            - 1.25 * final_distance
            - 0.020 * effort
            - 0.12 * smoothness
            - blockage_penalty
            - fragility_penalty
            - slip_penalty
        )
        return RolloutResult(
            states=states,
            utility=float(np.clip(utility, -2.5, 1.6)),
            mode=mode,
            final_distance=final_distance,
            blockage_penalty=float(blockage_penalty),
            fragility_penalty=float(fragility_penalty),
            slip_penalty=float(slip_penalty),
            effort=float(effort),
        )

    def free_mode_rollout(self, state: np.ndarray, actions: np.ndarray, goal: np.ndarray) -> RolloutResult:
        return self.rollout(state, actions, goal, mode="free", rng=np.random.default_rng(12345))

    def features(self, state: np.ndarray, actions: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.concatenate([state.reshape(-1), actions.reshape(-1), goal.reshape(-1)]).astype(np.float32)

    def simulate_dataset(self, num_samples: int, seed: int = 0) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = np.empty((num_samples, self.feature_dim), dtype=np.float32)
        y = np.empty((num_samples, self.target_dim), dtype=np.float32)
        rewards = np.empty(num_samples, dtype=np.float32)
        modes: list[str] = []
        for i in range(num_samples):
            cond = self.sample_condition(rng, condition_id=i)
            actions = self.sample_candidate_actions(cond.state, cond.goal, 1, rng)[0]
            rollout = self.rollout(cond.state, actions, cond.goal, cond.mode, rng=np.random.default_rng(seed * 1009 + i))
            x[i] = self.features(cond.state, actions, cond.goal)
            y[i] = rollout.states.reshape(-1)
            rewards[i] = rollout.utility
            modes.append(cond.mode)
        return {"features": x, "targets": y, "rewards": rewards, "modes": np.array(modes)}
