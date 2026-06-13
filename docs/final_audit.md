# Final Audit

Verification date: 2026-06-13.

## Command Results

| Command | Result | Runtime |
|---|---:|---:|
| `bash scripts/run_smoke.sh` | passed | script elapsed 100.07s; observed wall 169.2s |
| `bash scripts/run_all.sh` | passed | script elapsed 326.10s; observed wall 368.9s |
| `pytest` | passed, 22 tests | pytest 26.95s; observed wall 34.2s |
| `bash scripts/run_claim_audit.sh` | passed | observed wall 19.0s |

## Artifact Inventory

- Figures: `figures/figure1_tail_hallucination.png`, `figures/figure2_repair_comparison.png`, `figures/figure3_tail_diagnostics.png`, `figures/figure4_denoising_vs_selection.png`, `figures/figure5_exact_law_validation.png`, `figures/figure6_pilot_repair_gap_closure.png`, `figures/figure7_adaptive_n_gate.png`, `figures/figure8_calibration_reliability.png`, `figures/figure9_near_oracle_ablation.png`.
- Tables: `results/tables/main_metrics.csv`, `results/tables/seed_metrics.csv`, `results/tables/learned_metrics.csv`, `results/tables/denoising_grid.csv`, `results/tables/exact_law_validation.csv`, `results/tables/pilot_repair_metrics.csv`, `results/tables/gap_closure_by_budget.csv`, `results/tables/adaptive_n_metrics.csv`, `results/tables/calibration_diagnostics.csv`, `results/tables/learned_generalization_metrics.csv`, `results/tables/learned_training_curve.csv`.
- Model artifacts: `results/models/learned_diffusion_world_model.pt`, `results/models/learned_diffusion_world_model_ensemble0.pt`, `results/models/learned_diffusion_world_model_ensemble1.pt`, `results/models/learned_diffusion_world_model_ensemble2.pt`, `results/models/learned_training_summary.json`.
- Claim audit: `results/claims_status.md`, `results/claims_status.json`.

## Strongest Hallucination Artifact

Full-run controlled optimistic raw scoring at `N=64` has selected imagined score `1.375`, selected real utility `0.533`, high-`N` regret `0.133`, imagined-real tail gap `0.641`, and oracle gap `0.477`. The imagined score rises across `N`, while real utility peaks earlier and then drops.

Under the learned diffusion-world-model raw scorer at `N=64`, selected imagined score is `1.067`, selected real utility is `-0.754`, high-`N` regret is `0.896`, imagined-real tail gap is `1.460`, and the deployment gate is `block_high_n`.

## Pilot Repair Artifact

The practical held-out controlled pilot repair at `N=64` reports:

| Pilot budget | Raw real | Fixed real | Oracle real | Gap closed |
|---:|---:|---:|---:|---:|
| 0 | 0.846 | 0.960 | 1.116 | 42.2% |
| 8 | 0.846 | 1.091 | 1.116 | 90.5% |
| 32 | 0.846 | 1.092 | 1.116 | 90.9% |
| 128 | 0.846 | 1.091 | 1.116 | 90.5% |

The learned held-out pilot repair closes `94.1%` of the raw-to-oracle gap at budget `32`. The controlled upper-bound `repair_oracle_features` and `repair_many_pilot_labels` ablations each close `100.0%` of the gap and are explicitly labeled non-deployable.

## Learned Generalization Artifact

The learned ensemble uses three CPU-first denoisers. The primary member's training loss decreases from `0.296` to `0.133` over 5 epochs. Held-out generalization rows report test future-trajectory MSE `0.155`, final-state error `0.731`, held-out utility rank correlation `-0.393`, selected-tail calibration error `1.337`, and sample diversity `0.143`.

## Claim Boundary

Supported claim: pilot-label calibrated lower-confidence selection substantially reduces selected-tail hallucination and closes most of the oracle gap in controlled support-covered regimes.

Unsupported claims remain blocked: real-robot validation, broad robotics benchmark coverage, universal tail-selection repair, guaranteed 100% oracle recovery without additional information, and treating diffusion likelihood or imagined score as real utility.

The impossibility boundary is explicit in `docs/theory.md`: if two candidates have identical observable/generated features but different hidden real utility, feature-only selection cannot always choose the better candidate.

## Paper-Readiness Judgment

**paper-worthy controlled v1, needs benchmark validation for broader robotics claims.**

The controlled and learned toy evidence supports the scoped diagnostic and repair claims. The repo should not be presented as robot-planning validation or large-benchmark evidence.
