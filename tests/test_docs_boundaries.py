from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_and_paper_do_not_make_robotics_or_universal_repair_claims():
    paths = [ROOT / "README.md"] + sorted((ROOT / "paper").glob("*.md"))
    text = "\n".join(p.read_text(encoding="utf-8").lower() for p in paths if p.exists())
    forbidden_assertions = [
        "we solve robot planning",
        "we validate on real robots",
        "best-of-n always helps",
        "more samples always hurt",
        "calibration always fixes",
        "diffusion likelihood equals real utility",
    ]
    for phrase in forbidden_assertions:
        assert phrase not in text


def test_required_docs_exist_and_mark_scope_boundaries():
    for rel in [
        "docs/theory.md",
        "docs/claims.md",
        "docs/differentiation_from_best_of_n_wam.md",
        "docs/differentiation_from_prior_projects.md",
        "docs/reviewer_attacks.md",
        "docs/final_audit.md",
    ]:
        assert (ROOT / rel).exists()
    claims = (ROOT / "docs" / "claims.md").read_text(encoding="utf-8")
    assert "UNSUPPORTED" in claims
    assert "real robots" in claims
