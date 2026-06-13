# Abstract Notes

Tail selection can amplify model errors when selection acts on the upper tail of imagined future scores. This project studies that effect in diffusion world models that generate future trajectories conditioned on state, action sequence, and goal. In controlled toy worlds, selected imagined score can rise with `N` while real utility stagnates or decreases. We use an exact finite tie-aware top-tail law as a measurement tool and evaluate calibrated, uncertainty-aware, and consistency-aware scoring as scoped repairs.
