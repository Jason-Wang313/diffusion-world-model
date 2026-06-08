import json
import subprocess
from pathlib import Path

from dwm_best_of_n.audit import STATUSES, write_claim_audit


ROOT = Path(__file__).resolve().parents[1]


def _ensure_artifacts():
    needed = ROOT / "figures" / "figure1_tail_hallucination.png"
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
