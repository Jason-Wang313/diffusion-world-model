"""Controlled Best-of-N diffusion world-model experiments."""

from .evaluation import N_VALUES, deployment_gate
from .theory import finite_best_of_n_law, finite_best_of_n_curve

__all__ = [
    "N_VALUES",
    "deployment_gate",
    "finite_best_of_n_law",
    "finite_best_of_n_curve",
]
