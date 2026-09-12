"""Verify that the repository contains only readable thesis figure assets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "assets" / "figures"
EXPECTED = (
    "thesis_figure_1_proposed_workflow.jpg",
    "thesis_figure_2_confusion_matrix.png",
)


def main() -> None:
    actual = tuple(sorted(path.name for path in FIGURES.glob("*")))
    expected = tuple(sorted(EXPECTED))
    if actual != expected:
        raise SystemExit(
            f"Unexpected figure assets. Expected {expected}, found {actual}."
        )

    for name in EXPECTED:
        path = FIGURES / name
        with Image.open(path) as image:
            image.verify()
        print(f"verified: {path}")


if __name__ == "__main__":
    main()
