# Codex Handoff

## Current Artifact

- Repo: `C:\Users\wangz\diffusion world model`
- Remote: `https://github.com/Jason-Wang313/diffusion-world-model.git`
- Branch: `main`
- Final repo PDF: `paper/final/diffusion world model-v4.pdf`
- Final Desktop PDF: `C:\Users\wangz\OneDrive\Desktop\diffusion world model-v4.pdf`

## Verification Commands

```bash
python scripts/build_v4_paper.py
python scripts/run_v4_claim_audit.py
pytest
```

## Evidence Cache

- Frozen v4 cache: `results/v4_frozen_evidence/`
- Paper figures: `figures/v4/`
- Macros: `paper_iclr/v4_results_macros.tex`
- Benchmark tables: `v4_benchmark_candidates.csv`, `v4_benchmark_selection_curves.csv`, `v4_benchmark_law_validation.csv`, `v4_benchmark_summary.csv`

## Claim Boundaries

Supported: controlled selected-tail hallucination, learned toy denoiser stress, exact finite-pool audit, support-covered pilot repair, and Gymnasium Classic Control replay-pool baselines.

Unsupported: real-robot validation, SOTA controller performance, broad robotics benchmark coverage, universal repair, and treating imagined score or diffusion likelihood as real utility.
