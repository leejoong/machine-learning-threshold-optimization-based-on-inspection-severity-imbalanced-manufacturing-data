"""Generate the public result charts from the thesis test-result CSV files."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "paper_tables"
OUTPUT_DIR = ROOT / "assets" / "figures"

METHODS = ["0.5", "F2최적화", "OMMA", "제안"]
METHOD_LABELS = ["0.5", "F2 optimization", "OMMA", "Proposed"]
MODELS = ["XGBoost", "LightGBM", "CatBoost"]
MODEL_COLORS = {
    "XGBoost": "#2563EB",
    "LightGBM": "#0F766E",
    "CatBoost": "#D97706",
}
DATASETS = {
    "rsw_gun": ("RSW Gun", "rsw_gun_test_results.csv"),
    "wm811k": ("WM811K Wafer Map", "wm811k_test_results.csv"),
    "cnc_milling_tool_life": (
        "CNC Milling Tool Life",
        "cnc_milling_tool_life_test_results.csv",
    ),
}


def plot_dataset(dataset_key: str, title: str, csv_name: str) -> None:
    results = pd.read_csv(TABLE_DIR / csv_name)
    results["method"] = pd.Categorical(
        results["method"], categories=METHODS, ordered=True
    )
    results["model"] = pd.Categorical(
        results["model"], categories=MODELS, ordered=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    x = np.arange(len(METHODS))
    width = 0.22

    for ax, metric in zip(axes, ["f2", "gmean"]):
        for model_index, model in enumerate(MODELS):
            model_rows = results[results["model"] == model].set_index("method")
            values = [float(model_rows.loc[method, metric]) for method in METHODS]
            bars = ax.bar(
                x + (model_index - 1) * width,
                values,
                width,
                label=model,
                color=MODEL_COLORS[model],
                edgecolor="#111827",
                linewidth=0.35,
            )
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

        ax.axvspan(2.62, 3.38, color="#EFF6FF", zorder=0)
        ax.set_xticks(x, METHOD_LABELS)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(metric.upper() if metric == "f2" else "G-mean")
        ax.set_title("F2 score" if metric == "f2" else "G-mean")
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            3,
            1.04,
            "Proposed",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1D4ED8",
            fontweight="bold",
        )

    axes[0].legend(
        loc="upper left",
        frameon=False,
        ncols=3,
        bbox_to_anchor=(0, 1.16),
    )
    fig.suptitle(f"{title}: held-out test result comparison", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Source: thesis Tables 2–10 and the structured test-result CSVs | Higher is better",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    fig.savefig(
        OUTPUT_DIR / f"thesis_result_comparison_{dataset_key}.png",
        dpi=220,
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_key, (title, csv_name) in DATASETS.items():
        plot_dataset(dataset_key, title, csv_name)


if __name__ == "__main__":
    main()
