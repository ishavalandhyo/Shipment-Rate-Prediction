"""Create a compact visual overview of input-data quality and distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize freight input-data findings.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="scorer_results/input_data_overview.png")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = pd.read_csv(data_dir / "train_test.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    figure.suptitle("Freight Rate Input Data: Exploration and Quality Findings", fontsize=16, fontweight="bold", x=0.05, ha="left")

    # Missing values in each supplied dataset.
    missing = pd.DataFrame({
        "Training": [train["weight"].isna().sum(), train["market_index"].isna().sum()],
        "Validation": [validation["weight"].isna().sum(), validation["market_index"].isna().sum()],
    }, index=["weight", "market_index"])
    missing.plot.bar(ax=axes[0], color=["#0B7285", "#74C0FC"], width=0.7)
    axes[0].set_title("Missing numeric values", fontweight="bold")
    axes[0].set_ylabel("Rows")
    axes[0].set_xlabel("")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    # Weight quality before the documented correction.
    weight_quality = pd.DataFrame({
        "Training": [train["weight"].lt(0).sum(), train["weight"].isna().sum()],
        "Validation": [validation["weight"].lt(0).sum(), validation["weight"].isna().sum()],
    }, index=["Negative", "Missing"])
    weight_quality.plot.bar(ax=axes[1], color=["#2F9E44", "#E03131"], width=0.7)
    axes[1].set_title("Weight-quality exceptions before cleaning", fontweight="bold")
    axes[1].set_ylabel("Rows")
    axes[1].set_xlabel("")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)

    figure.text(0.05, 0.01, "Training: 48,000 rows | Validation: 12,000 rows", fontsize=10, color="#495057")
    figure.tight_layout(rect=(0.03, 0.04, 1, 0.90))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    print(f"Created input-data overview: {output}")


if __name__ == "__main__":
    main()
