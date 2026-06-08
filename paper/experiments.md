# Experiment Notes

Experiment A uses controlled diffusion-like generators: good, optimistic, mode-collapsed, and plausibility-biased.

Experiment B trains a CPU-first ensemble of three small conditional denoising MLPs on toy trajectories, samples future states, and reports held-out trajectory error, final-state error, utility rank correlation, selected-tail calibration error, diversity, and denoising-loss proxies.

Experiment C uses hidden slip, blockage, and fragility modes to expose multimodal future mismatch.

Experiment D varies denoising budget `K` and selection budget `N`.

Experiment E compares raw scoring with calibrated, uncertainty-aware, consistency-aware, random, and oracle scoring.

Experiment F evaluates pilot-label calibrated lower-confidence repair at budgets `0`, `8`, `32`, and `128`, reporting raw-to-oracle gap closure.

Experiment G evaluates adaptive `N` gates with reason codes and near-oracle controlled upper-bound ablations with hidden hazard features or many labels.
