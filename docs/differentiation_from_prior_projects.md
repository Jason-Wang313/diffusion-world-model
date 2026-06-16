# Differentiation From Prior Projects

## Diffusion Policy

Diffusion Policy generates action trajectories and may rank or condition them for control. This repository studies diffusion-generated **future-world trajectories** conditioned on state, action sequence, and goal. The selected object is a scored generated future/action candidate, and real utility is measured by the toy environment.

## JEPA / Latent Reranking

Latent reranking studies ranking in representation space and latent-real distortion. This repository uses explicit generated future states and measures imagined score against real rollout utility.

## WAM

The WAM connection is theoretical: finite top-tail laws apply to any score/utility pool. The empirical focus here is diffusion-world-model selection, especially generated future tails, denoising budget, mode collapse, plausibility bias, and uncertainty-aware repair.

## Robot Benchmarks

V4 implements standard Gymnasium Classic Control replay-pool audits for candidate-level selected-tail evidence. This is deliberately not a robot benchmark, not a SOTA controller claim, and not broad robotics coverage. The current contribution is a controlled diffusion-world diagnostic plus lightweight benchmark replay-pool evidence with explicit anti-overclaim boundaries.
