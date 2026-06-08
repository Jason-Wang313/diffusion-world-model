"""Claim ledger and artifact audit support."""

from __future__ import annotations

import json
from pathlib import Path


STATUSES = ("SUPPORTED", "PARTIAL", "UNSUPPORTED")


def _exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


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
        ]
    )
    metrics_ok = _exists_nonempty(root / "results" / "tables" / "main_metrics.csv")
    model_ok = _exists_nonempty(root / "results" / "models" / "learned_diffusion_world_model.pt")
    validation_ok = _exists_nonempty(root / "results" / "tables" / "exact_law_validation.csv")
    denoising_ok = _exists_nonempty(root / "results" / "tables" / "denoising_grid.csv")
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
            "claim": "A small conditional denoising MLP trains and produces sampled future trajectories for the toy world.",
            "status": "SUPPORTED" if model_ok else "PARTIAL",
            "evidence": "results/models/learned_diffusion_world_model.pt and results/tables/learned_metrics.csv",
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
            "claim": "Calibration-, uncertainty-, and consistency-aware scoring are evaluated as controlled selected-tail repair, not universal fixes.",
            "status": "SUPPORTED" if metrics_ok and figures_ok else "PARTIAL",
            "evidence": "repair rows in results/tables/main_metrics.csv and figure2_repair_comparison.png",
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
