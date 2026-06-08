from dwm_best_of_n.evaluation import (
    GATE_DECISIONS,
    GATE_REASONS,
    N_VALUES,
    deployment_gate,
    evaluate_analytic_variant,
)


def test_deployment_gate_outputs_allowed_values():
    assert deployment_gate(0.3, 0.0, 1.4, 0.4) in GATE_DECISIONS
    assert deployment_gate(0.0, 0.8, 0.1, 0.0) == "allow_high_n"


def test_metric_schema_from_tiny_controlled_eval():
    metrics, seeds = evaluate_analytic_variant(
        "controlled",
        "optimistic",
        ["raw", "calibrated"],
        seeds=[0],
        n_conditions=3,
        ns=(1, 2, 4),
        denoising_steps=4,
    )
    required = {
        "experiment",
        "generator",
        "scorer",
        "N",
        "selected_imagined_score_mean",
        "selected_real_utility_mean",
        "upper_tail_rank_correlation",
        "imagined_real_tail_gap",
        "high_n_regret",
        "oracle_gap",
        "exact_law_prediction_error",
        "deployment_gate",
        "generated_future_diversity",
    }
    assert required.issubset(metrics.columns)
    assert set(metrics["deployment_gate"]).issubset(set(GATE_DECISIONS))
    assert set(metrics["gate_reason"]).issubset(set(GATE_REASONS))
    assert len(seeds) == 2 * 3
    assert max(N_VALUES) == 64
