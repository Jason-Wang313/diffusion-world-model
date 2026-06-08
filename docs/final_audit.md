# Final Audit

Verification date: 2026-06-08.

## Command Results

| Command | Result | Runtime |
|---|---:|---:|
| `bash scripts/run_smoke.sh` | passed | script elapsed 147.57s; observed wall 202.5s |
| `bash scripts/run_all.sh` | passed | script elapsed 449.95s; observed wall 491.8s |
| `bash scripts/run_claim_audit.sh` | passed | observed wall 11.5s after documentation update |
| `pytest` | passed, 13 tests | pytest 43.12s; observed wall 55.3s after documentation update |

## Artifact Inventory

- Figures: `figures/figure1_tail_hallucination.png`, `figures/figure2_repair_comparison.png`, `figures/figure3_tail_diagnostics.png`, `figures/figure4_denoising_vs_selection.png`, `figures/figure5_exact_law_validation.png`.
- Tables: `results/tables/main_metrics.csv`, `results/tables/seed_metrics.csv`, `results/tables/learned_metrics.csv`, `results/tables/denoising_grid.csv`, `results/tables/exact_law_validation.csv`, `results/tables/learned_training_curve.csv`.
- Model artifacts: `results/models/learned_diffusion_world_model.pt`, `results/models/learned_training_summary.json`.
- Claim audit: `results/claims_status.md`, `results/claims_status.json`.

## Strongest Hallucination Artifact

Full-run controlled optimistic raw scoring at `N=64` has selected imagined score `1.375`, selected real utility `0.533`, high-`N` regret `0.133`, imagined-real tail gap `0.641`, and oracle gap `0.477`. The imagined score rises across `N`, while real utility peaks earlier and then drops.

Mode-collapsed raw scoring is similar: selected imagined score `1.561`, selected real utility `0.564`, high-`N` regret `0.138`, and imagined-real tail gap `0.841`.

## Strongest Learned Diffusion-World-Model Artifact

The learned denoising MLP trained with a clean-target denoising objective. Full-run training loss decreased from `0.292` to `0.150` over 5 epochs. Under raw scoring at `N=64`, selected imagined score is `0.980`, selected real utility is `-0.803`, high-`N` regret is `0.945`, imagined-real tail gap is `1.379`, and the deployment gate is `block_high_n`.

## Strongest Repair Artifact

At `N=64` in the optimistic repair experiment, raw selected real utility is `0.533`. Calibrated scoring reaches `0.594`, uncertainty-aware scoring reaches `0.589`, and consistency-aware scoring reaches `0.569`. These repairs lower imagined score and reduce the tail gap, but they are not universal fixes: the gate still returns `stop_early` or `collect_pilot_labels` where high-`N` regret remains.

## Diffusion-World-Model Validity Checklist

- Conditional future generation uses state, action sequence, goal, and denoising timestep.
- Samples are future-state trajectories, not action-only policies.
- Real utility is measured separately by the toy world rollout.
- Denoising budget `K` and selection budget `N` are separately varied.
- Learned-model artifacts are saved and covered by smoke tests.

## WAM Differentiation Checklist

- The exact finite law is reused as an abstract measurement tool, not claimed as a novel theorem.
- Experiments focus on diffusion-generated future trajectories.
- Metrics include imagined-real tail gap, upper-tail rank correlation, generated future diversity, denoising-vs-selection, and deployment gates.
- Documentation explicitly distinguishes WAM's imagined rollout mismatch from generated-future selected-tail hallucination.

## Not-Clone Checklist

- No WAM code or experiment files were copied.
- The repo has its own toy world, diffusion-style learned model, analytic future generators, scorer suite, audit ledger, and figures.
- Prior-project boundaries are documented in `docs/differentiation_from_prior_projects.md`.

## Paper-Readiness Judgment

**paper-worthy controlled v1, needs benchmark validation for broader robotics claims.**

The controlled and learned toy evidence strongly supports the scoped claims. The repo should not be presented as robot-planning validation or large-benchmark evidence.

## Top Remaining Weaknesses

- The environment is diagnostic and low-dimensional.
- The learned model is deliberately small and CPU-first.
- Repair methods improve selected-tail behavior but still require conservative gates.
- External robotics benchmarks are unsupported in v1.

## Exact Next Steps

- Add a benchmark adapter only after defining real utility independently of internal model score.
- Train a larger diffusion world model with held-out calibration labels.
- Add uncertainty calibration curves and pilot-label ablations.
- Preserve the current claim audit boundaries until benchmark evidence exists.
