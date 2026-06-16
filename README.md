# When More Dreaming Hurts

This repository studies **selection-budget tail audits for diffusion world models**: a model generates future-world trajectories from a state, action sequence, and goal; an internal score chooses the highest-scored generated future from a budget `N`; and real utility is measured separately by the toy environment.

The core thesis is selected-tail hallucination. As `N` grows, the chosen imagined future can look better while the selected real utility stagnates or drops, because the upper tail of generated futures is not calibrated to the upper tail of real outcomes. The repair claim is scoped: pilot-label calibrated lower-confidence selection can close most of the oracle gap in controlled support-covered regimes, while high `N` is blocked when the tail remains unidentifiable.

## Final v4 Submission Package

The current submission artifact is `paper/final/diffusion world model-v4.pdf`, copied to the Desktop as `diffusion world model-v4.pdf`. The v4 verification path is RAM-light: it regenerates the frozen evidence cache, runs a small Gymnasium Classic Control replay-pool benchmark audit, compiles the ICLR paper, and checks the Desktop/source-map package.

```bash
python scripts/build_v4_paper.py
python scripts/run_v4_claim_audit.py
pytest
```

## How This Differs

- **World-action models (WAM):** WAM focuses on imagined rollout versus real dynamics mismatch. This repo focuses on selection over diffusion-generated future samples and the behavior of generated-future tails under denoising quality, mode collapse, plausibility bias, and calibration.
- **Diffusion Policy:** Diffusion Policy generates action trajectories. This repo keeps the spotlight on generated future-world trajectories and the gap between their internal score and externally measured utility.
- **JEPA / latent reranking:** Latent rank distortion is related, but this repo measures imagined future score against real rollout utility in a concrete world-model selection loop.

## Quickstart

```bash
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/run_claim_audit.sh
python scripts/build_v4_paper.py
python scripts/run_v4_claim_audit.py
pytest
```

The scripts set `PYTHONPATH=src` and write CSV/JSON artifacts under `results/`, with key figures copied into `figures/`.

## Key Figures

- `figures/figure1_tail_hallucination.png`: selected imagined score versus selected real utility as `N` increases.
- `figures/figure2_repair_comparison.png`: raw scoring versus calibrated, uncertainty-aware, consistency-aware, random, and oracle scoring.
- `figures/figure3_tail_diagnostics.png`: imagined-real tail gap and upper-tail rank correlation.
- `figures/figure4_denoising_vs_selection.png`: denoising budget `K` versus selection budget `N`.
- `figures/figure5_exact_law_validation.png`: exact finite tie-aware top-tail law versus Monte Carlo.
- `figures/figure6_pilot_repair_gap_closure.png`: gap closure by pilot-label budget.
- `figures/figure7_adaptive_n_gate.png`: adaptive selection-budget gate decisions.
- `figures/figure8_calibration_reliability.png`: residual lower-bound calibration diagnostics.
- `figures/figure9_near_oracle_ablation.png`: controlled upper-bound ablations.
- `figures/v4/v4_gymnasium_benchmark_baselines.pdf`: Gymnasium benchmark replay-pool baselines.
- `figures/v4/v4_gymnasium_benchmark_deltas.pdf`: benchmark deltas against random with stress controls.

## Key Results

The full run uses 4 seeds, 24 toy conditions per seed, `N = {1,2,4,8,16,32,64}`, and 25k Monte Carlo trials for the exact-law check.

- Controlled optimistic raw scoring at `N=64`: selected imagined score `1.375`, selected real utility `0.533`, high-`N` regret `0.133`, imagined-real tail gap `0.641`.
- Learned raw scoring at `N=64`: selected imagined score `1.048`, selected real utility `-0.782`, high-`N` regret `0.924`, imagined-real tail gap `1.447`, deployment gate `block_high_n`.
- Learned denoising ensemble losses decrease over 5 epochs; the primary member decreases from `0.300` to `0.141`.
- Pilot repair at `N=64`: on held-out controlled optimistic conditions, budget `32` closes `90.9%` of the raw-to-oracle gap and budget `128` closes `90.5%`.
- Learned pilot repair at `N=64`: budget `32` closes `91.8%` of the raw-to-oracle gap on held-out learned diffusion-world-model conditions.
- Gymnasium Classic Control replay-pool audit: 3 tasks, 18 held-out pools, 1,152 candidate futures, 4 learned-score/LCB rows with positive lower confidence bounds over random, and one anti-score negative-control row below random.
- Near-oracle hidden-hazard and many-label upper-bound ablations each close `100.0%` of the held-out controlled gap.
- Adaptive gates emit exactly one of `allow_high_n`, `stop_early`, `collect_pilot_labels`, or `block_high_n` with a reason code.
- Exact finite-law validation max absolute Monte Carlo error: `0.0028`.

The learned component is intentionally small: a CPU-friendly conditional denoising MLP trained on toy trajectories. The benchmark layer is candidate-pool replay on standard Gymnasium simulators, not SOTA controller training or robotics validation.

## Claim Boundaries

This is a scoped v4 selected-tail audit. It supports controlled diagnostics, learned toy-denoiser stress tests, support-covered repair, and standard Gymnasium replay-pool baselines. It does not claim real-robot validation, SOTA controller performance, broad robotics benchmark coverage, or a universal repair method. Near-oracle ablations are controlled upper bounds, not deployable repairs. The exact claim ledger is generated at `results/claims_status.md` and `results/claims_status.json`.

For the final verified state, see `docs/final_audit.md`.
