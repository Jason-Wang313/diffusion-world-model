# Experiment Notes

Experiment A uses controlled diffusion-like generators: good, optimistic, mode-collapsed, and plausibility-biased.

Experiment B trains a small conditional denoising MLP on toy trajectories and samples future states.

Experiment C uses hidden slip, blockage, and fragility modes to expose multimodal future mismatch.

Experiment D varies denoising budget `K` and selection budget `N`.

Experiment E compares raw scoring with calibrated, uncertainty-aware, consistency-aware, random, and oracle scoring.
