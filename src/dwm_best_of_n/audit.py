"""Claim ledger and artifact audit support."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STATUSES = ("SUPPORTED", "PARTIAL", "UNSUPPORTED")


def _exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _gap_value(root: Path, experiment: str, repair_model: str, budget: int) -> float | None:
    path = root / "results" / "tables" / "gap_closure_by_budget.csv"
    if not _exists_nonempty(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    rows = df[
        (df["experiment"] == experiment)
        & (df["repair_model"] == repair_model)
        & (df["pilot_budget"] == budget)
    ]
    if rows.empty:
        return None
    return float(rows.iloc[0]["gap_closed"])


def build_claims(repo_root: str | Path = ".") -> list[dict[str, str]]:
    root = Path(repo_root)
    figures_ok = all(
        _exists_nonempty(root / "figures" / name)
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
        ]
    )
    metrics_ok = _exists_nonempty(root / "results" / "tables" / "main_metrics.csv")
    model_ok = _exists_nonempty(root / "results" / "models" / "learned_diffusion_world_model.pt")
    validation_ok = _exists_nonempty(root / "results" / "tables" / "exact_law_validation.csv")
    denoising_ok = _exists_nonempty(root / "results" / "tables" / "denoising_grid.csv")
    pilot_ok = _exists_nonempty(root / "results" / "tables" / "pilot_repair_metrics.csv")
    adaptive_ok = _exists_nonempty(root / "results" / "tables" / "adaptive_n_metrics.csv")
    calibration_ok = _exists_nonempty(root / "results" / "tables" / "calibration_diagnostics.csv")
    learned_gen_ok = _exists_nonempty(root / "results" / "tables" / "learned_generalization_metrics.csv")
    controlled_gap32 = _gap_value(root, "controlled_pilot_repair", "pilot_lcb", 32)
    controlled_gap128 = _gap_value(root, "controlled_pilot_repair", "pilot_lcb", 128)
    learned_gap32 = _gap_value(root, "learned_pilot_repair", "pilot_lcb", 32)
    oracle_gap128 = _gap_value(root, "near_oracle_ablation", "repair_oracle_features", 128)
    many_label_gap512 = _gap_value(root, "near_oracle_ablation", "repair_many_pilot_labels", 512)
    controlled_repair_supported = controlled_gap32 is not None and controlled_gap32 >= 0.70
    controlled_repair_partial = controlled_gap32 is not None and controlled_gap32 > 0.0
    learned_repair_supported = learned_gap32 is not None and learned_gap32 >= 0.70
    learned_repair_partial = learned_gap32 is not None and learned_gap32 >= 0.50
    near_oracle_supported = (
        oracle_gap128 is not None
        and oracle_gap128 >= 0.95
        and many_label_gap512 is not None
        and many_label_gap512 >= 0.95
    )
    claims = [
        {
            "group": "theorem claims",
            "claim": "Finite tie-aware Best-of-N expected selected utility is implemented and Monte Carlo validated.",
            "status": "SUPPORTED" if validation_ok else "PARTIAL",
            "evidence": "results/tables/exact_law_validation.csv and figure5_exact_law_validation.png",
        },
        {
            "group": "controlled toy claims",
            "claim": "A controlled diffusion-world toy shows selected imagined score rising while selected real utility can stagnate or drop.",
            "status": "SUPPORTED" if metrics_ok and figures_ok else "PARTIAL",
            "evidence": "results/tables/main_metrics.csv and figure1_tail_hallucination.png",
        },
        {
            "group": "learned diffusion-world-model claims",
            "claim": "A 3-member small conditional denoising ensemble trains and is evaluated on held-out toy conditions.",
            "status": "SUPPORTED" if model_ok and learned_gen_ok else "PARTIAL",
            "evidence": "results/models/learned_diffusion_world_model.pt and results/tables/learned_generalization_metrics.csv",
        },
        {
            "group": "multimodal/mode-collapse claims",
            "claim": "Mode-collapsed and hidden-mode generators expose selected-tail rank distortion and diversity diagnostics.",
            "status": "SUPPORTED" if metrics_ok else "PARTIAL",
            "evidence": "controlled rows for mode_collapsed and figure3_tail_diagnostics.png",
        },
        {
            "group": "denoising-budget claims",
            "claim": "Selection budget N and denoising budget K are separated in a CPU-controlled grid.",
            "status": "SUPPORTED" if denoising_ok else "PARTIAL",
            "evidence": "results/tables/denoising_grid.csv and figure4_denoising_vs_selection.png",
        },
        {
            "group": "repair claims",
            "claim": "Pilot-label calibrated lower-confidence selection closes most of the oracle gap in the controlled optimistic toy at budget 32.",
            "status": "SUPPORTED" if controlled_repair_supported else ("PARTIAL" if controlled_repair_partial else "UNSUPPORTED"),
            "evidence": "results/tables/gap_closure_by_budget.csv and figure6_pilot_repair_gap_closure.png",
        },
        {
            "group": "repair claims",
            "claim": "Budget 128 pilot repair is a stronger controlled repair than the budget 32 practical setting.",
            "status": "SUPPORTED" if controlled_gap128 is not None and controlled_gap128 >= 0.85 else "PARTIAL",
            "evidence": "pilot_budget=128 rows in results/tables/gap_closure_by_budget.csv",
        },
        {
            "group": "repair claims",
            "claim": "Adaptive Best-of-N deployment emits one gate decision with an explicit reason code.",
            "status": "SUPPORTED" if adaptive_ok else "PARTIAL",
            "evidence": "results/tables/adaptive_n_metrics.csv and figure7_adaptive_n_gate.png",
        },
        {
            "group": "calibration claims",
            "claim": "Residual conformal calibration reports held-out lower-bound diagnostics.",
            "status": "SUPPORTED" if calibration_ok else "PARTIAL",
            "evidence": "results/tables/calibration_diagnostics.csv and figure8_calibration_reliability.png",
        },
        {
            "group": "learned repair claims",
            "claim": "Pilot repair closes at least 70% of the oracle gap on held-out learned diffusion-world-model conditions.",
            "status": "SUPPORTED" if learned_repair_supported else ("PARTIAL" if learned_repair_partial else "UNSUPPORTED"),
            "evidence": "learned_pilot_repair rows in results/tables/gap_closure_by_budget.csv",
        },
        {
            "group": "near-oracle upper-bound claims",
            "claim": "Near-100% closure is possible in the controlled toy when hidden hazard features or enough labels are supplied.",
            "status": "SUPPORTED" if near_oracle_supported else "PARTIAL",
            "evidence": "repair_oracle_features rows and figure9_near_oracle_ablation.png",
        },
        {
            "group": "optional benchmark claims",
            "claim": "External robotics or benchmark validation is implemented.",
            "status": "UNSUPPORTED",
            "evidence": "intentionally out of scope for v1",
        },
        {
            "group": "unsupported future robotics claims",
            "claim": "The project solves robot planning or validates on real robots.",
            "status": "UNSUPPORTED",
            "evidence": "blocked by README, docs/claims.md, and docs/final_audit.md",
        },
        {
            "group": "forbidden overclaims",
            "claim": "Best-of-N always helps; more samples always hurt; calibration always fixes the issue; diffusion likelihood equals real utility.",
            "status": "UNSUPPORTED",
            "evidence": "blocked claim boundaries in docs/claims.md",
        },
        {
            "group": "forbidden overclaims",
            "claim": "Universal 100% Best-of-N repair is guaranteed without additional information.",
            "status": "UNSUPPORTED",
            "evidence": "blocked by hidden-mode impossibility note in docs/theory.md",
        },
        {
            "group": "forbidden overclaims",
            "claim": "This is just a renamed WAM project.",
            "status": "UNSUPPORTED",
            "evidence": "docs/differentiation_from_best_of_n_wam.md and diffusion-world-specific experiments",
        },
    ]
    return claims


def write_claim_audit(repo_root: str | Path = ".") -> tuple[Path, Path]:
    root = Path(repo_root)
    claims = build_claims(root)
    for item in claims:
        if item["status"] not in STATUSES:
            raise ValueError(f"bad status {item['status']}")
    json_path = root / "results" / "claims_status.json"
    md_path = root / "results" / "claims_status.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"claims": claims}, indent=2) + "\n", encoding="utf-8")
    lines = ["# Claim Status", "", "| Group | Status | Claim | Evidence |", "|---|---:|---|---|"]
    for item in claims:
        lines.append(f"| {item['group']} | {item['status']} | {item['claim']} | {item['evidence']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    json_path, md_path = write_claim_audit(Path.cwd())
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
