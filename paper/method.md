# Method Notes

The method separates three quantities:

- generated future `hat{x}_{1:H}`;
- imagined score `S`;
- externally measured real utility `R`.

Selection uses `argmax_i S_i`. Evaluation never treats the internal score as ground truth. Metrics include selected imagined score, selected real utility, upper-tail rank correlation, imagined-real tail gap, high-`N` regret, oracle gap, diversity, exact-law prediction error, and a deployment gate.
