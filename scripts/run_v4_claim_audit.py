from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "OneDrive" / "Desktop"
FINAL_NAME = "diffusion world model-v4.pdf"
REPO_PDF = ROOT / "paper" / "final" / FINAL_NAME
DESKTOP_PDF = DESKTOP / FINAL_NAME
SOURCE_MAP = DESKTOP / "PAPER_SOURCE_MAP.md"
SUMMARY = ROOT / "results" / "v4_frozen_evidence" / "summary.json"
CLAIMS = ROOT / "results" / "claims_status.json"
LATEX_LOG = ROOT / "paper_iclr" / "main.log"

EXPECTED_CACHE_FILES = [
    ROOT / "results" / "v4_frozen_evidence" / "summary.json",
    ROOT / "results" / "v4_frozen_evidence" / "v4_artifact_inventory.csv",
    ROOT / "results" / "v4_frozen_evidence" / "v4_benchmark_candidates.csv",
    ROOT / "results" / "v4_frozen_evidence" / "v4_benchmark_law_validation.csv",
    ROOT / "results" / "v4_frozen_evidence" / "v4_benchmark_selection_curves.csv",
    ROOT / "results" / "v4_frozen_evidence" / "v4_benchmark_summary.csv",
    ROOT / "results" / "v4_frozen_evidence" / "v4_calibration_diagnostics.csv",
    ROOT / "results" / "v4_frozen_evidence" / "v4_claim_inventory.csv",
    ROOT / "results" / "v4_frozen_evidence" / "v4_denoising_grid.csv",
    ROOT / "results" / "v4_frozen_evidence" / "v4_n64_tail_failures.csv",
    ROOT / "results" / "v4_frozen_evidence" / "v4_repair_budget.csv",
    ROOT / "results" / "v4_frozen_evidence" / "v4_seed_robustness.csv",
    ROOT / "paper_iclr" / "v4_results_macros.tex",
]

EXPECTED_FIGURES = [
    "v4_calibration_coverage.pdf",
    "v4_claim_artifact_inventory.pdf",
    "v4_denoising_selection_heatmap.pdf",
    "v4_gymnasium_benchmark_baselines.pdf",
    "v4_gymnasium_benchmark_deltas.pdf",
    "v4_repair_budget_curve.pdf",
    "v4_seed_robustness.pdf",
    "v4_tail_failure_landscape.pdf",
]

LOG_BLOCKERS = [
    r"! LaTeX Error",
    r"Undefined control sequence",
    r"Emergency stop",
    r"Fatal error",
    r"Counter too large",
    r"Overfull",
    r"Citation .*undefined",
    r"Reference .*undefined",
    r"There were undefined",
    r"Label\(s\) may have changed",
    r"Rerun to get",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
        env=env,
    )


def load_json(path: Path) -> dict[str, Any]:
    require(path.exists(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_pages(path: Path) -> int:
    require(path.exists(), f"missing PDF: {path}")
    output = run(["pdfinfo", str(path)]).stdout
    match = re.search(r"^Pages:\s+(\d+)", output, re.MULTILINE)
    require(match is not None, f"could not read page count for {path}")
    return int(match.group(1))


def check_cache_files() -> None:
    for path in EXPECTED_CACHE_FILES:
        require(path.exists(), f"missing v4 cache file: {path}")
    for name in EXPECTED_FIGURES:
        require((ROOT / "results" / "v4_frozen_evidence" / "figures" / name).exists(), f"missing result figure {name}")
        require((ROOT / "figures" / "v4" / name).exists(), f"missing paper figure {name}")


def check_summary(summary: dict[str, Any]) -> None:
    require(summary.get("supported_claims") >= 12, "supported claim count regressed")
    require(summary.get("partial_claims") == 0, "partial claims present")
    require(summary.get("unsupported_boundary_claims") >= 5, "boundary claims missing")
    require(summary.get("result_table_files") >= 11, "result table count regressed")
    require(summary.get("seed_rows") >= 336, "seed-row count regressed")
    require(summary.get("main_metric_rows") >= 84, "main metric row count regressed")
    require(summary.get("denoising_rows") >= 28, "denoising grid row count regressed")
    require(summary.get("pilot_repair_rows") >= 9, "pilot repair row count regressed")
    require(summary.get("adaptive_gate_rows") >= 63, "adaptive gate row count regressed")
    require(summary.get("calibration_rows") >= 9, "calibration row count regressed")
    require(summary.get("exact_law_max_abs_error", 1.0) <= 0.003, "exact-law error too high")
    require(summary.get("optimistic_raw_n64_imagined", 0.0) > 1.3, "optimistic imagined score too small")
    require(summary.get("optimistic_raw_n64_real", 1.0) < 0.6, "optimistic real utility no longer shows failure")
    require(summary.get("optimistic_raw_n64_tail_gap", 0.0) > 0.6, "optimistic tail gap too small")
    require(summary.get("optimistic_raw_n64_high_regret", 0.0) > 0.1, "optimistic high-N regret too small")
    require(summary.get("mode_collapsed_n64_tail_gap", 0.0) > 0.8, "mode-collapsed tail gap too small")
    require(summary.get("plausibility_biased_n64_tail_gap", 0.0) > 1.0, "plausibility tail gap too small")
    require(summary.get("learned_raw_n64_real", 1.0) < 0.0, "learned raw failure disappeared")
    require(summary.get("learned_raw_n64_tail_gap", 0.0) > 1.3, "learned raw tail gap too small")
    require(summary.get("learned_raw_n64_high_regret", 0.0) > 0.8, "learned raw regret too small")
    require(summary.get("learned_cal_n64_real", 0.0) > 0.5, "learned calibrated utility regressed")
    require(summary.get("controlled_gap_closed_budget32", 0.0) > 0.9, "controlled budget-32 repair regressed")
    require(summary.get("learned_gap_closed_budget32", 0.0) > 0.9, "learned budget-32 repair regressed")
    require(summary.get("training_loss_last", 1.0) < summary.get("training_loss_first", 0.0), "training loss did not decrease")
    require(summary.get("benchmark_envs") == 3, "benchmark env count regressed")
    require(summary.get("benchmark_eval_pools", 0) >= 18, "benchmark eval-pool count regressed")
    require(summary.get("benchmark_candidate_rows", 0) >= 1100, "benchmark candidate rows regressed")
    require(summary.get("benchmark_curve_rows", 0) >= 1100, "benchmark curve rows regressed")
    require(summary.get("benchmark_positive_ci_rows", 0) >= 4, "benchmark positive-CI rows regressed")
    require(summary.get("benchmark_anti_negative_rows", 0) >= 1, "benchmark anti-scorer negative controls missing")
    require(summary.get("benchmark_law_max_abs_error", 1.0) < 0.02, "benchmark exact-law error too high")


def check_claims() -> None:
    completed = run([sys.executable, "-m", "dwm_tail_audit.audit"], check=False)
    if completed.returncode != 0:
        print(completed.stdout)
        fail("claim audit module failed")
    payload = load_json(CLAIMS)
    claims = payload.get("claims") or []
    supported = sum(1 for claim in claims if claim.get("status") == "SUPPORTED")
    partial = sum(1 for claim in claims if claim.get("status") == "PARTIAL")
    unsupported = sum(1 for claim in claims if claim.get("status") == "UNSUPPORTED")
    require(supported >= 12, "claim audit supported count regressed")
    require(partial == 0, "claim audit has partial claims")
    require(unsupported >= 5, "claim audit lost unsupported boundary claims")


def check_source_map() -> None:
    require(SOURCE_MAP.exists(), f"missing source map: {SOURCE_MAP}")
    text = SOURCE_MAP.read_text(encoding="utf-8")
    expected = f"| `{FINAL_NAME}` | `{ROOT}` | `Jason-Wang313/diffusion-world-model` |"
    require(expected in text, "source map does not point diffusion world model to v4")
    require("diffusion world model-v2.pdf" not in text, "source map still contains diffusion world model v2")
    require("diffusion world model-v3.pdf" not in text, "source map still contains diffusion world model v3")


def check_latex_log() -> None:
    require(LATEX_LOG.exists(), "missing LaTeX log")
    text = LATEX_LOG.read_text(encoding="utf-8", errors="replace")
    blockers = [pattern for pattern in LOG_BLOCKERS if re.search(pattern, text, re.IGNORECASE)]
    require(not blockers, f"LaTeX log blockers present: {blockers}")


def check_git_tracking() -> None:
    tracked_build_pdf = run(["git", "ls-files", "--error-unmatch", "paper_iclr/main.pdf"], check=False)
    require(tracked_build_pdf.returncode != 0, "generated paper_iclr/main.pdf is still tracked")


def main() -> None:
    run([sys.executable, "experiments/v4_frozen_evidence.py"])
    check_cache_files()
    check_summary(load_json(SUMMARY))
    check_claims()

    repo_pages = pdf_pages(REPO_PDF)
    desktop_pages = pdf_pages(DESKTOP_PDF)
    require(repo_pages >= 25, f"repo final PDF has only {repo_pages} pages")
    require(desktop_pages >= 25, f"Desktop final PDF has only {desktop_pages} pages")
    require(sha256(REPO_PDF) == sha256(DESKTOP_PDF), "repo and Desktop PDFs differ")
    require(not (DESKTOP / "diffusion world model-v2.pdf").exists(), "stale Desktop v2 PDF exists")
    require(not (DESKTOP / "diffusion world model-v3.pdf").exists(), "stale Desktop v3 PDF exists")

    check_source_map()
    check_latex_log()
    check_git_tracking()

    print("Diffusion world model v4 audit passed")
    print(f"pages={repo_pages} sha256={sha256(REPO_PDF)}")


if __name__ == "__main__":
    main()
