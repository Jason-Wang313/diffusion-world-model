# Differentiation From Prior Projects

## Diffusion Policy

Diffusion Policy generates action trajectories and may rank or condition them for control. This repository studies diffusion-generated **future-world trajectories** conditioned on state, action sequence, and goal. The selected object is a scored generated future/action candidate, and real utility is measured by the toy environment.

## JEPA / Latent Reranking

Latent reranking studies ranking in representation space and latent-real distortion. This repository uses explicit generated future states and measures imagined score against real rollout utility.

## WAM

The WAM connection is theoretical: finite top-tail laws apply to any score/utility pool. The empirical focus here is diffusion-world-model selection, especially generated future tails, denoising budget, mode collapse, plausibility bias, and uncertainty-aware repair.

## Robot Benchmarks

Robot and large benchmark validation are not implemented in v1. The current contribution is a controlled, audit-friendly toy evidence package.
