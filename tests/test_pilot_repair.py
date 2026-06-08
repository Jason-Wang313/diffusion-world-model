import numpy as np

from dwm_best_of_n.evaluation import GATE_DECISIONS, GATE_REASONS, make_analytic_batches
from dwm_best_of_n.pilot_repair import (
    condition_splits,
    conformal_lower_quantile,
    fit_pilot_repair,
    gap_closure_metrics,
    run_pilot_repair_experiment,
)


def _tiny_groups():
    return [
        (seed, make_analytic_batches("optimistic", seed, n_conditions=5, max_n=8, denoising_steps=4))
        for seed in [0, 1]
    ]


def test_gap_closure_metric_correctness_and_zero_gap_edge_case():
    metrics = gap_closure_metrics(raw_real=0.0, fixed_real=0.5, oracle_real=1.0)
    assert abs(metrics["gap_closed"] - 0.5) < 1e-12
    zero = gap_closure_metrics(raw_real=1.0, fixed_real=1.0, oracle_real=1.0)
    assert zero["oracle_gap_raw"] == 0.0
    assert zero["gap_closed"] == 0.0


def test_pilot_condition_splits_do_not_leak_eval_conditions():
    split = condition_splits(_tiny_groups(), seed=7)
    assert split.train_keys
    assert split.calibration_keys
    assert split.eval_keys
    assert split.train_keys.isdisjoint(split.calibration_keys)
    assert split.train_keys.isdisjoint(split.eval_keys)
    assert split.calibration_keys.isdisjoint(split.eval_keys)


def test_conformal_lower_bound_monotonic_with_confidence_level():
    true = np.array([0.0, -0.2, -0.3, 0.1, -0.6])
    pred = np.array([0.1, 0.1, -0.1, 0.2, 0.0])
    q50 = conformal_lower_quantile(true, pred, confidence=0.50)
    q90 = conformal_lower_quantile(true, pred, confidence=0.90)
    assert q90 >= q50


def test_adaptive_gate_has_exactly_one_decision_and_valid_reason():
    metrics, _, adaptive, _ = run_pilot_repair_experiment(
        _tiny_groups(),
        budgets=(0, 8),
        ns=(1, 2, 4),
        split_seed=3,
    )
    for frame in [metrics, adaptive]:
        decision_sum = frame[list(GATE_DECISIONS)].sum(axis=1)
        assert (decision_sum == 1).all()
        assert set(frame["deployment_gate"]).issubset(set(GATE_DECISIONS))
        assert set(frame["gate_reason"]).issubset(set(GATE_REASONS))


def test_near_oracle_ablation_is_labeled_controlled_upper_bound():
    metrics, gap, _, _ = run_pilot_repair_experiment(
        _tiny_groups(),
        budgets=(8,),
        ns=(1, 2, 4),
        experiment="near_oracle_ablation",
        repair_model="repair_oracle_features",
        include_oracle_features=True,
        split_seed=5,
    )
    assert metrics["controlled_upper_bound"].all()
    assert not metrics["deployable_repair"].any()
    assert gap["controlled_upper_bound"].all()


def test_fit_pilot_repair_keeps_eval_conditions_unlabeled():
    fitted = fit_pilot_repair(_tiny_groups(), budget=8, split_seed=9)
    assert fitted.train_label_count > 0
    assert fitted.calibration_label_count > 0
    assert fitted.split.train_keys.isdisjoint(fitted.split.eval_keys)
    assert fitted.split.calibration_keys.isdisjoint(fitted.split.eval_keys)
