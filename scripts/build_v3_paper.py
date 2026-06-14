from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper_iclr"
TEX = PAPER_DIR / "main.tex"
PDF = PAPER_DIR / "main.pdf"
FINAL_NAME = "diffusion world model-v3.pdf"
FINAL_PDF = ROOT / "paper" / "final" / FINAL_NAME
DESKTOP_PDF = Path.home() / "OneDrive" / "Desktop" / FINAL_NAME


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    env = os.environ.copy()
    env.setdefault("SOURCE_DATE_EPOCH", "1700000000")
    env.setdefault("FORCE_SOURCE_DATE", "1")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def main() -> None:
    run(["python", "experiments/v3_cached_evidence.py"])
    for suffix in [".aux", ".bbl", ".blg", ".log", ".out", ".pdf"]:
        target = PAPER_DIR / f"main{suffix}"
        if target.exists():
            target.unlink()

    run(["pdflatex", "-interaction=nonstopmode", TEX.name], cwd=PAPER_DIR)
    run(["bibtex", "main"], cwd=PAPER_DIR)
    for _ in range(4):
        run(["pdflatex", "-interaction=nonstopmode", TEX.name], cwd=PAPER_DIR)

    FINAL_PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PDF, FINAL_PDF)
    shutil.copy2(PDF, DESKTOP_PDF)
    print(f"PDF: {DESKTOP_PDF}")
    print(f"Repo PDF: {FINAL_PDF}")


if __name__ == "__main__":
    main()
