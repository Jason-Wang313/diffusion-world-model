# Claim Status

| Group | Status | Claim | Evidence |
|---|---:|---|---|
| theorem claims | SUPPORTED | Finite tie-aware top-tail expected selected utility is implemented and Monte Carlo validated. | results/tables/exact_law_validation.csv and figure5_exact_law_validation.png |
| controlled toy claims | SUPPORTED | A controlled diffusion-world toy shows selected imagined score rising while selected real utility can stagnate or drop. | results/tables/main_metrics.csv and figure1_tail_hallucination.png |
| learned diffusion-world-model claims | SUPPORTED | A 3-member small conditional denoising ensemble trains and is evaluated on held-out toy conditions. | results/models/learned_diffusion_world_model.pt and results/tables/learned_generalization_metrics.csv |
| multimodal/mode-collapse claims | SUPPORTED | Mode-collapsed and hidden-mode generators expose selected-tail rank distortion and diversity diagnostics. | controlled rows for mode_collapsed and figure3_tail_diagnostics.png |
| denoising-budget claims | SUPPORTED | Selection budget N and denoising budget K are separated in a CPU-controlled grid. | results/tables/denoising_grid.csv and figure4_denoising_vs_selection.png |
| repair claims | SUPPORTED | Pilot-label calibrated lower-confidence selection closes most of the oracle gap in the controlled optimistic toy at budget 32. | results/tables/gap_closure_by_budget.csv and figure6_pilot_repair_gap_closure.png |
| repair claims | SUPPORTED | Budget 128 pilot repair is a stronger controlled repair than the budget 32 practical setting. | pilot_budget=128 rows in results/tables/gap_closure_by_budget.csv |
| repair claims | SUPPORTED | Adaptive selection-budget deployment emits one gate decision with an explicit reason code. | results/tables/adaptive_n_metrics.csv and figure7_adaptive_n_gate.png |
| calibration claims | SUPPORTED | Residual conformal calibration reports held-out lower-bound diagnostics. | results/tables/calibration_diagnostics.csv and figure8_calibration_reliability.png |
| learned repair claims | SUPPORTED | Pilot repair closes at least 70% of the oracle gap on held-out learned diffusion-world-model conditions. | learned_pilot_repair rows in results/tables/gap_closure_by_budget.csv |
| near-oracle upper-bound claims | SUPPORTED | Near-100% closure is possible in the controlled toy when hidden hazard features or enough labels are supplied. | repair_oracle_features rows and figure9_near_oracle_ablation.png |
| optional benchmark claims | SUPPORTED | A standard Gymnasium Classic Control rollout-pool benchmark audit is implemented with random, learned-score, LCB, anti-score, and oracle baselines. | results/v4_frozen_evidence/v4_benchmark_summary.csv and v4_benchmark_selection_curves.csv |
| unsupported future robotics claims | UNSUPPORTED | SOTA controller performance, real-robot validation, or broad robotics benchmark coverage is implemented. | v4 benchmark evidence is candidate-pool replay on Gymnasium Classic Control, not robotics/SOTA control |
| unsupported future robotics claims | UNSUPPORTED | The project solves robot planning or validates on real robots. | blocked by README, docs/claims.md, and docs/final_audit.md |
| forbidden overclaims | UNSUPPORTED | Tail selection always helps; more samples always hurt; calibration always fixes the issue; diffusion likelihood equals real utility. | blocked claim boundaries in docs/claims.md |
| forbidden overclaims | UNSUPPORTED | Universal 100% tail-selection repair is guaranteed without additional information. | blocked by hidden-mode impossibility note in docs/theory.md |
| forbidden overclaims | UNSUPPORTED | This is just a renamed WAM project. | docs/differentiation_from_wam.md and diffusion-world-specific experiments |
