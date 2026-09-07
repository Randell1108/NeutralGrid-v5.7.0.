"""Render the retained development comparison using isolated Matplotlib."""
from __future__ import annotations

import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT / "_tmp/plot_lib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    results = json.loads((OUT / "development_results.json").read_text())
    selected = json.loads((OUT / "frozen_selection.json").read_text())["development"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for kind, color in [("logistic", "#2463A8"), ("vote", "#B55A2A")]:
        for live, style in [(False, "-"), (True, "--")]:
            rows = sorted([r for r in results if r["estimator"] == kind and r["profile"] == "rank" and r["live"] == live], key=lambda r: r["k"])
            label = kind + (" + 87 observed positives" if live else "")
            axes[0].plot([r["k"] for r in rows], [r["mean_fold_auc"] for r in rows], style, color=color, label=label)
            axes[1].plot([r["k"] for r in rows], [r["pooled"]["ece"] for r in rows], style, color=color)
    axes[0].scatter([selected["k"]], [selected["mean_fold_auc"]], marker="*", s=170, color="#222222", label="Frozen selection", zorder=5)
    axes[0].set(xlabel="Number of recorded features", ylabel="Mean development AUC", title="Five chronological development folds")
    axes[1].set(xlabel="Number of recorded features", ylabel="Pooled development ECE", title="Calibration error (lower is better)")
    axes[1].axhline(.1, color="#777777", linewidth=1, linestyle=":")
    axes[0].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle("FASTWIN: feature count and observed-positive augmentation", fontsize=13)
    fig.savefig(OUT / "feature_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
