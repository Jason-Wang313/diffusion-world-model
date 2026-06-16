# Claim Ledger Boundaries

Generated claim status lives in `results/claims_status.md` and `results/claims_status.json`. Each claim is one of `SUPPORTED`, `PARTIAL`, or `UNSUPPORTED`.

## Supported Or Partial Claim Families

- The finite tie-aware top-tail law is implemented for binary and real-valued utilities.
- Monte Carlo validates the exact law on finite tied pools.
- Controlled toy worlds show selected-tail hallucination for diffusion-like future generators.
- The small learned denoising MLP trains and samples future trajectories in the toy world.
- Mode collapse, plausibility bias, and denoising-budget effects are measured with diagnostics.
- Pilot-label calibrated lower-confidence selection is evaluated in the controlled setting.
- Adaptive selection-budget gates report `allow_high_n`, `stop_early`, `collect_pilot_labels`, or `block_high_n` with reason codes.
- Near-oracle ablations are controlled upper bounds when hidden hazard information is supplied.
- V4 implements a standard Gymnasium Classic Control replay-pool audit with random, learned-score, lower-confidence, anti-score, and oracle baselines.

## Explicitly Unsupported Claims

| Claim | Status | Boundary |
|---|---:|---|
| We solve robot planning. | UNSUPPORTED | No robot planner is implemented. |
| We validate on real robots. | UNSUPPORTED | No real-robot data or hardware experiment is present. |
| We achieve SOTA controller performance. | UNSUPPORTED | The Gymnasium evidence is candidate-pool replay, not controller training. |
| We provide broad robotics benchmark coverage. | UNSUPPORTED | V4 uses lightweight Classic Control replay pools, not robot task families. |
| Tail selection always helps. | UNSUPPORTED | The repo shows conditional behavior. |
| More samples always hurt. | UNSUPPORTED | The failure is selected-tail dependent, not universal. |
| Calibration always fixes the issue. | UNSUPPORTED | Repair is empirical and scoped. |
| Pilot labels guarantee 100% oracle recovery. | UNSUPPORTED | Hidden modes can be unidentifiable from generated features. |
| Diffusion likelihood equals real utility. | UNSUPPORTED | The repo separates imagined score from real utility. |
| This is only toy evidence. | UNSUPPORTED | V4 also has standard Gymnasium replay-pool evidence, but it remains scoped. |
| This is just a renamed WAM project. | UNSUPPORTED | The artifact studies diffusion-generated future tails. |
