"""Small conditional denoising MLP used as a CPU-first diffusion world model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class DiffusionConfig:
    condition_dim: int
    target_dim: int
    hidden_dim: int = 64
    timesteps: int = 16
    lr: float = 1e-3
    batch_size: int = 64
    device: str = "cpu"


class ConditionalDenoiser(nn.Module):
    def __init__(self, condition_dim: int, target_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim + target_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, noisy_target: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([condition, noisy_target, t.float()], dim=1))


class DiffusionWorldModel:
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = ConditionalDenoiser(config.condition_dim, config.target_dim, config.hidden_dim).to(self.device)
        betas = torch.linspace(1e-4, 0.035, config.timesteps, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def train_model(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int = 4,
        seed: int = 0,
    ) -> list[float]:
        torch.manual_seed(seed)
        x = torch.as_tensor(features, dtype=torch.float32)
        y = torch.as_tensor(targets, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(x, y),
            batch_size=min(self.config.batch_size, len(x)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        losses: list[float] = []
        self.model.train()
        for _ in range(epochs):
            epoch_losses = []
            for cond, clean in loader:
                cond = cond.to(self.device)
                clean = clean.to(self.device)
                idx = torch.randint(0, self.config.timesteps, (clean.shape[0],), device=self.device)
                alpha_bar = self.alpha_bars[idx][:, None]
                noise = torch.randn_like(clean)
                noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
                t_scaled = idx.float() / max(self.config.timesteps - 1, 1)
                pred_clean = self.model(noisy, cond, t_scaled)
                loss = torch.mean((pred_clean - clean) ** 2)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(epoch_losses)))
        return losses

    @torch.no_grad()
    def sample(
        self,
        condition: np.ndarray,
        n_samples: int = 1,
        steps: int | None = None,
        seed: int = 0,
    ) -> np.ndarray:
        cond_np = np.asarray(condition, dtype=np.float32).reshape(1, -1)
        cond_matrix = np.repeat(cond_np, n_samples, axis=0)
        return self.sample_conditions(cond_matrix, steps=steps, seed=seed)

    @torch.no_grad()
    def sample_conditions(
        self,
        conditions: np.ndarray,
        steps: int | None = None,
        seed: int = 0,
    ) -> np.ndarray:
        steps = int(steps or self.config.timesteps)
        steps = max(1, min(steps, self.config.timesteps))
        cond_np = np.asarray(conditions, dtype=np.float32)
        if cond_np.ndim != 2 or cond_np.shape[1] != self.config.condition_dim:
            raise ValueError(
                f"conditions must have shape (n, {self.config.condition_dim}), got {cond_np.shape}"
            )
        n_samples = int(cond_np.shape[0])
        generator = torch.Generator(device=self.device).manual_seed(seed)
        cond = torch.as_tensor(cond_np, dtype=torch.float32, device=self.device)
        x = torch.randn((n_samples, self.config.target_dim), generator=generator, device=self.device)
        self.model.eval()
        schedule = torch.linspace(self.config.timesteps - 1, 0, steps, device=self.device).round().long().unique(sorted=True)
        schedule = torch.flip(schedule, dims=(0,))
        for idx in schedule:
            t_idx = int(idx.item())
            t = torch.full((n_samples,), t_idx / max(self.config.timesteps - 1, 1), device=self.device)
            pred_clean = self.model(x, cond, t)
            beta = self.betas[t_idx]
            if t_idx > 0:
                x = 0.55 * x + 0.45 * pred_clean
                x = x + 0.45 * torch.sqrt(beta) * torch.randn(x.shape, generator=generator, device=self.device)
            else:
                x = pred_clean
        return x.detach().cpu().numpy().astype(np.float32)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.__dict__,
                "state_dict": self.model.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DiffusionWorldModel":
        payload = torch.load(path, map_location="cpu")
        model = cls(DiffusionConfig(**payload["config"]))
        model.model.load_state_dict(payload["state_dict"])
        return model
