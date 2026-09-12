"""Verify that thesis figures and result-table artifacts are readable."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "assets" / "figures"
PAPER_TABLES = ROOT / "results" / "paper_tables"
EXPECTED = (
    "thesis_figure_1_proposed_workflow.jpg",
    "thesis_figure_2_confusion_matrix.png",
    "thesis_result_comparison_rsw_gun.png",
    "thesis_result_comparison_wm811k.png",
    "thesis_result_comparison_cnc_milling_tool_life.png",
)
EXPECTED_TABLE_IMAGES = tuple(
    f"thesis_table_{index:02d}_" + suffix
    for index, suffix in [
        (1, "variables.png"),
        (2, "rsw_gun_05_vs_proposed.png"),
        (3, "rsw_gun_f2_vs_proposed.png"),
        (4, "rsw_gun_omma_vs_proposed.png"),
        (5, "wm811k_05_vs_proposed.png"),
        (6, "wm811k_f2_vs_proposed.png"),
        (7, "wm811k_omma_vs_proposed.png"),
        (8, "cnc_milling_tool_life_05_vs_proposed.png"),
        (9, "cnc_milling_tool_life_f2_vs_proposed.png"),
        (10, "cnc_milling_tool_life_omma_vs_proposed.png"),
    ]
)
EXPECTED_RESULT_CSVS = (
    "rsw_gun_test_results.csv",
    "wm811k_test_results.csv",
    "cnc_milling_tool_life_test_results.csv",
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

    for name in EXPECTED_TABLE_IMAGES:
        path = PAPER_TABLES / name
        with Image.open(path) as image:
            image.verify()
        print(f"verified: {path}")

    required_columns = {
        "dataset", "model", "method", "threshold", "precision", "recall",
        "f1", "f2", "gmean", "fpr", "tp", "fp", "fn", "tn",
    }
    for name in EXPECTED_RESULT_CSVS:
        path = PAPER_TABLES / name
        result = pd.read_csv(path)
        if len(result) != 12 or not required_columns.issubset(result.columns):
            raise SystemExit(f"Unexpected result table schema: {path}")
        if result["dataset"].nunique() != 1 or result["method"].nunique() != 4:
            raise SystemExit(f"Unexpected dataset or method values: {path}")
        print(f"verified: {path}")


if __name__ == "__main__":
    main()
