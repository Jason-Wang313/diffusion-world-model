# Final Audit

Verification date: 2026-06-19.

## Current Final Package

- Final repository PDF: `paper/final/diffusion world model-v4.pdf`.
- Final Desktop PDF: `C:\Users\wangz\OneDrive\Desktop\diffusion world model-v4.pdf`.
- Final PDF SHA-256: `E8339C4F40BB6EC43E2E6065F7B251E5F037F1DB42B52BF949DA2932503FFFC0`.
- Final PDF pages: 27.
- Matching GitHub repository: `https://github.com/Jason-Wang313/diffusion-world-model.git`.
- Visual QA inspected rendered pages 1, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 22, 23, and 27.
- PDF annotation link audit found 76 annotations: 44 green citation boxes and 32 red internal-reference boxes, all using 1pt visible borders.

## Command Results

| Command | Result | Runtime |
|---|---:|---:|
| `python -m compileall src tests experiments scripts -q` | passed | current v4 pass |
| `python -m pytest -q` | passed, 22 tests | current v4 pass |
| `bash scripts/run_smoke.sh` | passed | script elapsed 100.07s; observed wall 169.2s |
| `bash scripts/run_all.sh` | passed | script elapsed 326.10s; observed wall 368.9s |
| `pytest` | passed, 22 tests | pytest 26.95s; observed wall 34.2s |
| `bash scripts/run_claim_audit.sh` | passed | observed wall 19.0s |
| `python scripts/build_v4_paper.py` | passed | regenerated frozen evidence and Desktop PDF on 2026-06-19 |
| `python scripts/run_v4_claim_audit.py` | passed | pages=27, SHA-256 `E8339C4F40BB6EC43E2E6065F7B251E5F037F1DB42B52BF949DA2932503FFFC0` |
| Final LaTeX log blocker scan | passed | no undefined citations/references, rerun warnings, overfull boxes, or fatal errors |
| Boxed-link rebuild from frozen artifacts | passed | repository and Desktop PDFs match; selected high-risk pages rendered at moderate DPI |

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

## V4 Acceptance Gates

- The frozen evidence summary reports 12 supported claims, 0 partial claims, and 5 explicit unsupported boundary claims.
- The v4 evidence layer contains 11 result tables, 336 seed rows, 84 main metric rows, 28 denoising-grid rows, 9 pilot-repair rows, 63 adaptive-gate rows, and 9 calibration rows.
- The Gymnasium replay-pool layer covers 3 standard environments, 18 held-out pools, 1,152 candidate rows, 1,134 curve rows, 4 positive-CI learned-score/LCB rows over random, and 1 anti-score negative-control row.
- The exact finite-law error is `0.0027724326261199`; the benchmark law error is `0.013900772709148301`.
- The repo and Desktop PDFs have identical SHA-256 hashes.
- The source map points to the v4 Desktop PDF, this local folder, and `Jason-Wang313/diffusion-world-model`.
- Old v2/v3 Desktop diffusion PDFs are absent.
- Representative rendered pages passed visual QA.

## Claim Boundary

Supported claim: pilot-label calibrated lower-confidence selection substantially reduces selected-tail hallucination and closes most of the oracle gap in controlled support-covered regimes; the v4 audit also runs standard benchmark replay-pool baselines.

Unsupported claims remain blocked: real-robot validation, SOTA controller performance, broad robotics benchmark coverage, universal tail-selection repair, guaranteed 100% oracle recovery without additional information, and treating diffusion likelihood or imagined score as real utility.

The impossibility boundary is explicit in `docs/theory.md`: if two candidates have identical observable/generated features but different hidden real utility, feature-only selection cannot always choose the better candidate.

## Paper-Readiness Judgment

**submission-ready scoped v4, with benchmark replay-pool evidence and blocked robotics/SOTA claims.**

The controlled, learned toy, and Gymnasium replay-pool evidence supports the scoped diagnostic and repair claims. The repo should not be presented as robot-planning validation or SOTA controller evidence.
