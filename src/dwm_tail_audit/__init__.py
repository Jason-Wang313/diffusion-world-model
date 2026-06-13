"""Controlled diffusion-world-model tail audits."""

from .evaluation import N_VALUES, deployment_gate
from .theory import finite_top_tail_law, finite_top_tail_curve

__all__ = [
    "N_VALUES",
    "deployment_gate",
    "finite_top_tail_law",
    "finite_top_tail_curve",
]
