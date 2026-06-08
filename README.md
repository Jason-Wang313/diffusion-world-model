# When More Dreaming Hurts

This repository studies **Best-of-N inference for diffusion world models**: a model generates future-world trajectories from a state, action sequence, and goal; an internal score selects the best-looking future among `N`; and real utility is measured separately by the toy environment.

The core thesis is selected-tail hallucination. As `N` grows, the chosen imagined future can look better while the selected real utility stagnates or drops, because the upper tail of generated futures is not calibrated to the upper tail of real outcomes.

## How This Differs

- **Best-of-N WAM:** WAM focuses on imagined rollout versus real dynamics mismatch. This repo focuses on selection over diffusion-generated future samples and the behavior of generated-future tails under denoising quality, mode collapse, plausibility bias, and calibration.
- **Diffusion Policy:** Diffusion Policy generates action trajectories. This repo keeps the spotlight on generated future-world trajectories and the gap between their internal score and externally measured utility.
- **JEPA / latent Best-of-N:** Latent rank distortion is related, but this repo measures imagined future score against real rollout utility in a concrete world-model selection loop.

## Quickstart

```bash
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/run_claim_audit.sh
pytest
```

The scripts set `PYTHONPATH=src` and write CSV/JSON artifacts under `results/`, with key figures copied into `figures/`.

## Key Figures

- `figures/figure1_tail_hallucination.png`: selected imagined score versus selected real utility as `N` increases.
- `figures/figure2_repair_comparison.png`: raw scoring versus calibrated, uncertainty-aware, consistency-aware, random, and oracle scoring.
- `figures/figure3_tail_diagnostics.png`: imagined-real tail gap and upper-tail rank correlation.
- `figures/figure4_denoising_vs_selection.png`: denoising budget `K` versus selection budget `N`.
- `figures/figure5_exact_law_validation.png`: exact finite tie-aware Best-of-N law versus Monte Carlo.

## Key Results

The full run uses 4 seeds, 24 toy conditions per seed, `N = {1,2,4,8,16,32,64}`, and 25k Monte Carlo trials for the exact-law check.

- Controlled optimistic raw scoring at `N=64`: selected imagined score `1.375`, selected real utility `0.533`, high-`N` regret `0.133`, imagined-real tail gap `0.641`.
- Learned raw scoring at `N=64`: selected imagined score `0.980`, selected real utility `-0.803`, high-`N` regret `0.945`, imagined-real tail gap `1.379`, deployment gate `block_high_n`.
- Learned denoising loss decreases from `0.292` to `0.150` over 5 epochs.
- Repair at `N=64`: raw real utility `0.533`; calibrated `0.594`; uncertainty-aware `0.589`; consistency-aware `0.569`; oracle `1.010`.
- Exact finite-law validation max absolute Monte Carlo error: `0.0028`.

The learned component is intentionally small: a CPU-friendly conditional denoising MLP trained on toy trajectories. It is evidence that the pipeline handles learned diffusion-style future sampling, not a robotics benchmark.

## Claim Boundaries

This is a controlled research v1. It does not claim real-robot validation, broad benchmark coverage, or a universal repair method. The exact claim ledger is generated at `results/claims_status.md` and `results/claims_status.json`.

For the final verified state, see `docs/final_audit.md`.
