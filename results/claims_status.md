# Claim Status

| Group | Status | Claim | Evidence |
|---|---:|---|---|
| theorem claims | SUPPORTED | Finite tie-aware Best-of-N expected selected utility is implemented and Monte Carlo validated. | results/tables/exact_law_validation.csv and figure5_exact_law_validation.png |
| controlled toy claims | SUPPORTED | A controlled diffusion-world toy shows selected imagined score rising while selected real utility can stagnate or drop. | results/tables/main_metrics.csv and figure1_tail_hallucination.png |
| learned diffusion-world-model claims | SUPPORTED | A small conditional denoising MLP trains and produces sampled future trajectories for the toy world. | results/models/learned_diffusion_world_model.pt and results/tables/learned_metrics.csv |
| multimodal/mode-collapse claims | SUPPORTED | Mode-collapsed and hidden-mode generators expose selected-tail rank distortion and diversity diagnostics. | controlled rows for mode_collapsed and figure3_tail_diagnostics.png |
| denoising-budget claims | SUPPORTED | Selection budget N and denoising budget K are separated in a CPU-controlled grid. | results/tables/denoising_grid.csv and figure4_denoising_vs_selection.png |
| repair claims | SUPPORTED | Calibration-, uncertainty-, and consistency-aware scoring are evaluated as controlled selected-tail repair, not universal fixes. | repair rows in results/tables/main_metrics.csv and figure2_repair_comparison.png |
| optional benchmark claims | UNSUPPORTED | External robotics or benchmark validation is implemented. | intentionally out of scope for v1 |
| unsupported future robotics claims | UNSUPPORTED | The project solves robot planning or validates on real robots. | blocked by README, docs/claims.md, and docs/final_audit.md |
| forbidden overclaims | UNSUPPORTED | Best-of-N always helps; more samples always hurt; calibration always fixes the issue; diffusion likelihood equals real utility. | blocked claim boundaries in docs/claims.md |
| forbidden overclaims | UNSUPPORTED | This is just a renamed WAM project. | docs/differentiation_from_best_of_n_wam.md and diffusion-world-specific experiments |
