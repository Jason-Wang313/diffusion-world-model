# Differentiation From WAM

The shared piece is the abstract finite top-tail law: once a finite score/utility pool exists, the expected selected utility under top-score selection can be computed exactly with tie groups.

The object of study is different.

| Axis | WAM audit | This repo |
|---|---|---|
| Model object | Imagined rollout model | Diffusion world model over future trajectories |
| Failure mode | Rollout/real dynamics mismatch | Generated future tail hallucination |
| Selection target | Best imagined rollout | Best scored generated future/action candidate |
| Diagnostics | Utility law and mismatch curves | Tail gap, upper-tail rank correlation, diversity, mode frequency, denoising-vs-selection |
| Repair frame | General selection caution | Controlled selected-tail repair via calibration, uncertainty, and consistency |
| Scope | WAM-specific | Diffusion-world-model-specific controlled toy plus small learned DWM plus Gymnasium replay-pool audit |

The repo does not clone WAM experiments or claim the theorem is novel. It reuses the law as a measuring instrument and builds diffusion-world-model artifacts around generated future tails, denoising pressure, pilot calibration, and standard simulator replay-pool stress tests.
