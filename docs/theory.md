# Theory Notes

## Formal Setup

At decision time the agent observes state or observation `s_t` / `o_t`, a candidate action sequence `a_{t:t+H}`, and goal `g`. A diffusion world model defines a conditional future distribution:

```text
p_theta(future | state, action_sequence, goal)
```

A generated future is written `hat{x}_{1:H}`. The model or planner assigns an imagined score `S(hat{x}_{1:H}, a_{t:t+H}, g)`. The toy environment separately measures real utility `R` by replaying the selected action sequence under the real hidden mode. Best-of-N selection draws `N` generated futures or future/action candidates and selects:

```text
i* = argmax_i S_i
```

If energy notation is used, the score convention is `S = -E`; larger `S` is always preferred.

## Exact Finite Tie-Aware Best-of-N Law

For a finite empirical pool with scores `S_j` and utilities `R_j`, candidates are sampled iid uniformly. Sort tied score groups from highest to lowest. For score group `g`, let `p_g` be its pool probability, `u_g` its mean utility, and `P_higher(g)` the probability mass of all higher-score groups.

The probability that the top observed score belongs to group `g` after `N` samples is:

```text
(1 - P_higher(g))^N - (1 - P_higher(g) - p_g)^N
```

With uniform top-score tie handling, the conditional expected utility is `u_g`. Therefore:

```text
E[R_selected] = sum_g Pr(top group is g) u_g
```

This applies to real-valued utility and to binary utility. The binary case simply has `u_g` equal to the empirical success rate inside the tied score group.

## Edge Cases

- **Ties:** handled by tied score groups and uniform selection among sampled top-score ties.
- **Constant utility:** expected selected utility is constant for every `N`.
- **Oracle score:** if score is perfectly aligned with real utility, expected selected utility is nondecreasing with `N`.
- **Anti-aligned score:** if score is inversely aligned with real utility, expected selected utility can decrease with `N`.
- **Finite-pool expectation:** the law is exact for iid samples from the finite pool.
- **Monte Carlo validation:** `figure5_exact_law_validation.png` compares the closed-form law with simulated selection.

## Diffusion-Specific Additions

This repo uses the same abstract finite law, but studies diffusion-world-model failure modes:

- **Generative future tail hallucination:** the selected imagined future improves with `N` while selected real utility stagnates or drops.
- **Imagined-real tail gap:** the gap between upper-tail imagined score and corresponding real utility.
- **Upper-tail rank correlation:** rank agreement restricted to high imagined-score candidates.
- **High-N regret:** loss in selected real utility at large `N` relative to low-`N` selection.
- **Oracle-minus-diffusion gap:** the real utility gap between oracle selection and the evaluated scorer.
- **Plausibility-real mismatch:** plausible-looking generated futures need not have high real utility.
- **Mode collapse:** a model can sample the free/slip/stick mode while the hidden real mode is blocked or fragile.
- **Optimistic future bias:** generated futures can overstate goal progress for aggressive action sequences.
- **Denoising budget versus selection budget:** increasing `N` is separated from improving reverse-process quality `K`.

## WAM Distinction

WAM studies explicit imagined rollout / real dynamics mismatch. This repo studies selection over diffusion-generated futures and how generated-future tails behave under plausibility, denoising quality, multimodality, and calibration. The theorem is reused abstractly; the experiments, metrics, and claim boundaries are diffusion-world-model specific.
