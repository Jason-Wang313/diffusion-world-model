# Reviewer Attacks

## "This is just WAM with diffusion."

The exact finite law is reused as an abstract selection law, and the repo states that plainly. The new artifact is the diffusion-world-model setting: generated future trajectories, optimistic generated tails, denoising budget `K`, mode collapse, plausibility-real mismatch, diversity diagnostics, and controlled selected-tail repair.

## "This is just diffusion policy reranking."

Diffusion Policy centers on action-trajectory generation. This repo centers on future-world generation conditioned on state, action sequence, and goal. The evaluation asks whether the selected generated future's internal score is aligned with real rollout utility.

## "This is only a toy."

Yes. The claim is controlled toy evidence plus a small learned diffusion-world-model smoke artifact. The repo is deliberately not presented as robotics benchmark validation.

## "The theorem is reused."

Correct. The theorem is not claimed as novel. The law is used as a finite-pool measurement tool for diffusion-world-model selected-tail behavior.

## "Where is robotics?"

Not in v1. The repository avoids real-robot claims. A robotics extension would require benchmark integration, hardware or benchmark logs, policy interfaces, and new claim audit entries.
