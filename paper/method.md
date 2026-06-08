# Method Notes

The method separates three quantities:

- generated future `hat{x}_{1:H}`;
- imagined score `S`;
- externally measured real utility `R`.

Selection uses `argmax_i S_i` for the raw diagnostic. Evaluation never treats the internal score as ground truth. Metrics include selected imagined score, selected real utility, upper-tail rank correlation, imagined-real tail gap, high-`N` regret, oracle gap, gap closure, diversity, exact-law prediction error, and a deployment gate.

The repair path spends a small pilot-label budget on targeted real rollouts from high-score, high-uncertainty, low-consistency, disagreement, and random candidates. It fits a residual model for `R - S`, calibrates a one-sided residual quantile on held-out conditions, and selects by a lower-confidence predicted real utility. The adaptive deployment rule can allow high `N`, stop early, request pilot labels, or block high `N`, and every decision carries a reason code.
