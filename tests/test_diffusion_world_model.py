from pathlib import Path

import numpy as np

from dwm_best_of_n.diffusion_world_model import DiffusionConfig, DiffusionWorldModel
from dwm_best_of_n.toy_world import ToyWorld, ToyWorldConfig


def test_diffusion_world_model_training_and_sampling_shape(tmp_path: Path):
    world = ToyWorld(ToyWorldConfig(horizon=4))
    data = world.simulate_dataset(18, seed=4)
    model = DiffusionWorldModel(
        DiffusionConfig(
            condition_dim=world.feature_dim,
            target_dim=world.target_dim,
            hidden_dim=24,
            timesteps=5,
            batch_size=9,
        )
    )
    losses = model.train_model(data["features"], data["targets"], epochs=1, seed=4)
    assert len(losses) == 1
    samples = model.sample(data["features"][0], n_samples=3, steps=3, seed=5)
    assert samples.shape == (3, world.target_dim)
    assert np.isfinite(samples).all()
    path = tmp_path / "model.pt"
    model.save(path, metadata={"test": True})
    assert path.exists() and path.stat().st_size > 0
