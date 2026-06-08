import json
import subprocess
from pathlib import Path

import pandas as pd

from dwm_best_of_n.audit import STATUSES, write_claim_audit


ROOT = Path(__file__).resolve().parents[1]


def _ensure_artifacts():
    needed = ROOT / "figures" / "figure9_near_oracle_ablation.png"
    if not needed.exists() or needed.stat().st_size == 0:
        subprocess.run(["bash", "scripts/run_smoke.sh"], cwd=ROOT, check=True)


def test_required_figures_exist_and_are_nonempty_pngs():
    _ensure_artifacts()
    for name in [
        "figure1_tail_hallucination.png",
        "figure2_repair_comparison.png",
        "figure3_tail_diagnostics.png",
        "figure4_denoising_vs_selection.png",
        "figure5_exact_law_validation.png",
        "figure6_pilot_repair_gap_closure.png",
        "figure7_adaptive_n_gate.png",
        "figure8_calibration_reliability.png",
        "figure9_near_oracle_ablation.png",
    ]:
        path = ROOT / "figures" / name
        assert path.exists()
        assert path.stat().st_size > 1000
        assert path.read_bytes().startswith(b"\x89PNG")


def test_claim_audit_schema_and_forbidden_overclaim_statuses():
    _ensure_artifacts()
    json_path, _ = write_claim_audit(ROOT)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "claims" in payload and payload["claims"]
    for claim in payload["claims"]:
        assert claim["status"] in STATUSES
    forbidden = [c for c in payload["claims"] if c["group"] in {"forbidden overclaims", "unsupported future robotics claims"}]
    assert forbidden
    assert all(c["status"] == "UNSUPPORTED" for c in forbidden)
    universal = [c for c in forbidden if "Universal 100%" in c["claim"]]
    assert universal and universal[0]["status"] == "UNSUPPORTED"


def test_required_repair_tables_and_schemas_exist():
    _ensure_artifacts()
    expected = {
        "pilot_repair_metrics.csv": {
            "experiment",
            "generator",
            "repair_model",
            "pilot_budget",
            "N",
            "raw_real_utility",
            "fixed_real_utility",
            "oracle_real_utility",
            "gap_closed",
            "gate_reason",
        },
        "gap_closure_by_budget.csv": {
            "experiment",
            "repair_model",
            "pilot_budget",
            "oracle_gap_raw",
            "oracle_gap_fixed",
            "gap_closed",
            "controlled_upper_bound",
            "deployable_repair",
        },
        "adaptive_n_metrics.csv": {
            "allow_high_n",
            "stop_early",
            "collect_pilot_labels",
            "block_high_n",
            "deployment_gate",
            "gate_reason",
        },
        "calibration_diagnostics.csv": {
            "pilot_budget",
            "confidence",
            "conformal_quantile",
            "eval_lower_bound_coverage",
            "train_condition_count",
            "eval_condition_count",
        },
    }
    for filename, columns in expected.items():
        path = ROOT / "results" / "tables" / filename
        assert path.exists() and path.stat().st_size > 0
        df = pd.read_csv(path)
        assert columns.issubset(df.columns)


def test_adaptive_gate_table_has_one_decision_per_row():
    _ensure_artifacts()
    df = pd.read_csv(ROOT / "results" / "tables" / "adaptive_n_metrics.csv")
    decision_cols = ["allow_high_n", "stop_early", "collect_pilot_labels", "block_high_n"]
    assert (df[decision_cols].sum(axis=1) == 1).all()


def test_learned_generalization_table_has_required_schema():
    _ensure_artifacts()
    path = ROOT / "results" / "tables" / "learned_generalization_metrics.csv"
    assert path.exists() and path.stat().st_size > 0
    df = pd.read_csv(path)
    required = {
        "split",
        "future_trajectory_mse",
        "final_state_error",
        "utility_rank_correlation",
        "selected_tail_calibration_error",
        "sample_diversity",
        "negative_log_proxy",
        "denoising_loss_proxy",
        "ensemble_size",
    }
    assert required.issubset(df.columns)
