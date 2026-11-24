"""Headless plots of aggregate validation threshold metrics only."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

POLICY_COLORS = ("#64748b", "#0f766e", "#c46b12")


def save_threshold_charts(
    sweep: pd.DataFrame, comparison: pd.DataFrame, output_dir: Path, *, min_recall: float,
) -> None:
    labels = ("Baseline 0.5", "Max F1", f"Max precision, recall >= {min_recall:.0%}")
    figure = Figure(figsize=(10, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    for metric, color in (("precision", "#2563eb"), ("recall", "#b45309"), ("f1", "#0f766e")):
        axes.plot(sweep["threshold"], sweep[metric], label=metric.capitalize(), color=color, linewidth=1.8)
    for (_, row), label, color in zip(comparison.iterrows(), labels, POLICY_COLORS):
        axes.axvline(row["threshold"], color=color, linestyle="--", linewidth=1.2,
                     label=f"{label}: {row['threshold']:.4f}")
    axes.axhline(min_recall, color="#94a3b8", linestyle=":", linewidth=1)
    axes.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Decision threshold (predict fraud when score >= threshold)",
             ylabel="Metric", title="Validation threshold trade-offs | saved LightGBM, no refitting")
    axes.grid(alpha=0.18)
    axes.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    figure.savefig(output_dir / "threshold_metrics.png", dpi=160)

    figure = Figure(figsize=(9, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    # Exclude predict-none: precision=0 is a convention, not an observed alert.
    alerted = sweep.loc[sweep["flagged_rows"] > 0]
    axes.plot(alerted["recall"], alerted["precision"], color="#2563eb", linewidth=1.5, label="All score cutoffs")
    axes.axvspan(min_recall, 1, color="#e2e8f0", alpha=0.55, label="Recall constraint region")
    axes.axvline(min_recall, color="#64748b", linestyle=":", linewidth=1)
    for (_, row), label, color in zip(comparison.iterrows(), labels, POLICY_COLORS):
        axes.scatter(row["recall"], row["precision"], color=color, s=70, edgecolors="white", zorder=5,
                     label=f"{label} (t={row['threshold']:.4f})")
    axes.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
             title="Validation precision / recall | thresholds selected on this same sample")
    axes.grid(alpha=0.18)
    axes.legend(fontsize=8, loc="upper right")
    figure.savefig(output_dir / "threshold_precision_recall.png", dpi=160)

    figure = Figure(figsize=(13, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    for axis, (_, row), label in zip(axes, comparison.iterrows(), labels):
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        heatmap = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for actual in range(2):
            for predicted in range(2):
                axis.text(predicted, actual, f"{matrix[actual, predicted]:,}\n({normalized[actual, predicted]:.1%})",
                          ha="center", va="center", fontsize=11,
                          color="white" if normalized[actual, predicted] > 0.55 else "#172033")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-fraud", "Fraud"],
                 yticklabels=["Non-fraud", "Fraud"], xlabel="Predicted", ylabel="Actual",
                 title=f"{label}\nt = {row.threshold:.4f}")
    figure.colorbar(heatmap, ax=list(axes), shrink=0.75, label="Fraction of each actual class")
    figure.suptitle("Validation confusion matrices | counts and row-normalized percentages", fontsize=13)
    figure.savefig(output_dir / "threshold_confusion_matrices.png", dpi=160)
