# Final Audit

Verification date: 2026-06-16.

## Command Results

| Command | Result | Runtime |
|---|---:|---:|
| `bash scripts/run_smoke.sh` | passed | script elapsed 100.07s; observed wall 169.2s |
| `bash scripts/run_all.sh` | passed | script elapsed 326.10s; observed wall 368.9s |
| `pytest` | passed, 22 tests | pytest 26.95s; observed wall 34.2s |
| `bash scripts/run_claim_audit.sh` | passed | observed wall 19.0s |
| `python scripts/build_v4_paper.py` | passed | regenerates frozen evidence and Desktop PDF |
| `python scripts/run_v4_claim_audit.py` | passed | checks v4 PDF, benchmark gates, and source map |

## Artifact Inventory

- Figures: `figures/figure1_tail_hallucination.png`, `figures/figure2_repair_comparison.png`, `figures/figure3_tail_diagnostics.png`, `figures/figure4_denoising_vs_selection.png`, `figures/figure5_exact_law_validation.png`, `figures/figure6_pilot_repair_gap_closure.png`, `figures/figure7_adaptive_n_gate.png`, `figures/figure8_calibration_reliability.png`, `figures/figure9_near_oracle_ablation.png`.
- Tables: `results/tables/main_metrics.csv`, `results/tables/seed_metrics.csv`, `results/tables/learned_metrics.csv`, `results/tables/denoising_grid.csv`, `results/tables/exact_law_validation.csv`, `results/tables/pilot_repair_metrics.csv`, `results/tables/gap_closure_by_budget.csv`, `results/tables/adaptive_n_metrics.csv`, `results/tables/calibration_diagnostics.csv`, `results/tables/learned_generalization_metrics.csv`, `results/tables/learned_training_curve.csv`.
- Model artifacts: `results/models/learned_diffusion_world_model.pt`, `results/models/learned_diffusion_world_model_ensemble0.pt`, `results/models/learned_diffusion_world_model_ensemble1.pt`, `results/models/learned_diffusion_world_model_ensemble2.pt`, `results/models/learned_training_summary.json`.
- Claim audit: `results/claims_status.md`, `results/claims_status.json`.
- V4 frozen evidence: `results/v4_frozen_evidence/summary.json`, benchmark CSVs, and eight v4 figure PDFs under `figures/v4/`.

## Strongest Hallucination Artifact

Full-run controlled optimistic raw scoring at `N=64` has selected imagined score `1.375`, selected real utility `0.533`, high-`N` regret `0.133`, imagined-real tail gap `0.641`, and oracle gap `0.477`. The imagined score rises across `N`, while real utility peaks earlier and then drops.

Under the learned diffusion-world-model raw scorer at `N=64`, selected imagined score is `1.048`, selected real utility is `-0.782`, high-`N` regret is `0.924`, imagined-real tail gap is `1.447`, and the deployment gate is `block_high_n`.

## Pilot Repair Artifact

The practical held-out controlled pilot repair at `N=64` reports:

| Pilot budget | Raw real | Fixed real | Oracle real | Gap closed |
|---:|---:|---:|---:|---:|
| 0 | 0.846 | 0.960 | 1.116 | 42.2% |
| 8 | 0.846 | 1.091 | 1.116 | 90.5% |
| 32 | 0.846 | 1.092 | 1.116 | 90.9% |
| 128 | 0.846 | 1.091 | 1.116 | 90.5% |

The learned held-out pilot repair closes `91.8%` of the raw-to-oracle gap at budget `32`. The controlled upper-bound `repair_oracle_features` and `repair_many_pilot_labels` ablations each close `100.0%` of the gap and are explicitly labeled non-deployable.

## Learned Generalization Artifact

The learned ensemble uses three CPU-first denoisers. The primary member's training loss decreases from `0.300` to `0.141` over 5 epochs. Held-out generalization rows report test future-trajectory MSE `0.159`, final-state error `0.757`, held-out utility rank correlation `-0.407`, selected-tail calibration error `1.298`, and sample diversity `0.136`.

## Standard Benchmark Replay-Pool Artifact

V4 adds a Gymnasium Classic Control candidate-pool audit over CartPole-v1, Pendulum-v1, and MountainCarContinuous-v0. It contains 18 held-out pools, 1,152 candidate futures, random/learned-score/LCB/anti-score/oracle baselines, and exact-law validation with max absolute error below `0.014`. Four learned-score/LCB rows have positive lower confidence bounds over random; Pendulum remains a weak stress row, not a hidden failure.

## Claim Boundary

Supported claim: pilot-label calibrated lower-confidence selection substantially reduces selected-tail hallucination and closes most of the oracle gap in controlled support-covered regimes; the v4 audit also runs standard benchmark replay-pool baselines.

Unsupported claims remain blocked: real-robot validation, SOTA controller performance, broad robotics benchmark coverage, universal tail-selection repair, guaranteed 100% oracle recovery without additional information, and treating diffusion likelihood or imagined score as real utility.

The impossibility boundary is explicit in `docs/theory.md`: if two candidates have identical observable/generated features but different hidden real utility, feature-only selection cannot always choose the better candidate.

## Paper-Readiness Judgment

**submission-ready scoped v4, with benchmark replay-pool evidence and blocked robotics/SOTA claims.**

The controlled, learned toy, and Gymnasium replay-pool evidence supports the scoped diagnostic and repair claims. The repo should not be presented as robot-planning validation or SOTA controller evidence.
