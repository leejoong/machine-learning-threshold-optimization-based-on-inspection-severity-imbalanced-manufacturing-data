"""Build the public-facing figures from tracked summary tables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "assets" / "figures"


def _theme() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def build_performance() -> Path:
    table = pd.read_csv(TABLES / "journal_table_2_main_performance_average.csv")
    pivot = table.pivot(index="Dataset", columns="Method", values="F2")
    methods = [method for method in ["Fixed 0.5", "Global F2-only", "OMMA", "Proposed"] if method in pivot]
    ax = pivot[methods].plot(kind="bar", figsize=(9, 4.8), color=["#94a3b8", "#64748b", "#f59e0b", "#0f766e"])
    ax.set_title("F2 performance across manufacturing datasets")
    ax.set_xlabel("")
    ax.set_ylabel("F2 (higher is better)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="Decision policy", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    plt.tight_layout()
    path = FIGURES / "public_f2_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def build_risk_concentration() -> Path:
    table = pd.read_csv(TABLES / "journal_table_3_risk_concentration.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(table["Tightened row share (%)"], table["Positive capture share (%)"], s=90, color="#0f766e")
    for row in table.itertuples(index=False):
        ax.annotate(row.Dataset, (row[5], row[6]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.plot([0, 100], [0, 100], linestyle="--", color="#94a3b8", linewidth=1)
    ax.set_title("The policy concentrates inspection effort on risky rows")
    ax.set_xlabel("Rows under tightened inspection (%)")
    ax.set_ylabel("Positive rows captured (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path = FIGURES / "public_risk_concentration.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def build_cost() -> Path:
    table = pd.read_csv(TABLES / "journal_table_7_cost_sensitivity_fn10.csv")
    pivot = table.pivot(index="Dataset", columns="Method", values="Reduction vs fixed 0.5 (%)")
    methods = [method for method in ["Global F2-only", "OMMA", "Proposed"] if method in pivot]
    ax = pivot[methods].plot(kind="bar", figsize=(8.5, 4.8), color=["#64748b", "#f59e0b", "#0f766e"])
    ax.set_title("Expected inspection-cost reduction at FN:FP = 10:1")
    ax.set_xlabel("")
    ax.set_ylabel("Reduction vs fixed 0.5 (%)")
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="Decision policy", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    plt.tight_layout()
    path = FIGURES / "public_cost_sensitivity.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    _theme()
    for builder in (build_performance, build_risk_concentration, build_cost):
        print(builder())


if __name__ == "__main__":
    main()
