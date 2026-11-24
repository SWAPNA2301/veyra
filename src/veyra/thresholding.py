"""Optimize validation thresholds of a saved LightGBM pipeline; never fit it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .data import JOIN_KEY, TARGET_COLUMN, TIME_COLUMN
from .threshold_plots import save_threshold_charts
from .training import binary_metrics
from .training_data import load_development_data


def threshold_sweep(labels, probabilities) -> pd.DataFrame:
    """Exact O(n log n) threshold metrics, with tied scores kept together.

    Predictions use score >= threshold. Candidates include every distinct
    score, 0, 0.5, 1, and nextafter(max_score, +inf) for the predict-none state.
    That final sentinel may exceed 1 by one float step when a score equals 1.
    Undefined precision (no alerts) is zero. Both label classes are required.
    """
    labels = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Labels and scores must be matching nonempty 1-D arrays")
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Threshold optimization requires both binary classes 0/1")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    positive_prefix = np.concatenate(([0], np.cumsum(labels[order], dtype=np.int64)))
    positive_count = int(positive_prefix[-1])
    negative_count = len(labels) - positive_count
    thresholds = np.unique(np.concatenate((ordered_scores, [0.0, 0.5, 1.0, np.nextafter(ordered_scores[-1], np.inf)])))
    below = np.searchsorted(ordered_scores, thresholds, side="left")
    flagged = len(labels) - below
    tp = positive_count - positive_prefix[below]
    fp = flagged - tp
    fn, tn = positive_count - tp, negative_count - fp
    precision = np.divide(tp, flagged, out=np.zeros(len(thresholds)), where=flagged != 0)
    recall = tp / positive_count
    f1 = 2 * tp / (2 * tp + fp + fn)
    return pd.DataFrame({
        "threshold": thresholds, "precision": precision, "recall": recall, "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "flagged_rows": flagged, "flagged_pct": 100 * flagged / len(labels),
    })


def select_operating_points(sweep: pd.DataFrame, *, min_recall: float = 0.70) -> pd.DataFrame:
    """Compare fixed 0.5, max F1, and max precision satisfying the recall floor.

    Deterministic ties: higher recall, then higher precision (for max F1),
    then higher threshold. The recall constraint is never relaxed.
    """
    if not np.isfinite(min_recall) or not 0 <= min_recall <= 1:
        raise ValueError("min_recall must be in [0, 1]")
    if sweep.empty or 0.5 not in sweep["threshold"].values:
        raise ValueError("Threshold sweep must include baseline 0.5")
    feasible = sweep.loc[sweep["recall"] >= min_recall]
    if feasible.empty:
        raise ValueError("No threshold satisfies the minimum recall")
    baseline = sweep.loc[sweep["threshold"] == 0.5].head(1).copy()
    best_f1 = sweep.sort_values(["f1", "recall", "precision", "threshold"], ascending=False, kind="stable").head(1).copy()
    best_precision = feasible.sort_values(["precision", "recall", "threshold"], ascending=False, kind="stable").head(1).copy()
    parts = []
    for policy, frame in (("baseline_0_5", baseline), ("max_f1", best_f1), ("max_precision_at_recall", best_precision)):
        frame.insert(0, "policy", policy)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_baseline(baseline: dict, recorded: dict) -> None:
    for key in ("rows", "fraud_rows", "confusion_matrix", "threshold"):
        if baseline[key] != recorded[key]:
            raise ValueError(f"Validation baseline no longer matches saved training metrics: {key}")
    for key in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
        if not np.isclose(baseline[key], recorded[key], rtol=0, atol=1e-12):
            raise ValueError(f"Validation baseline no longer matches saved training metrics: {key}")


def optimize_saved_thresholds(
    raw_dir: Path, split_report: Path, training_metrics: Path, output_dir: Path,
    *, min_recall: float = 0.70,
) -> pd.DataFrame:
    """Only predict validation using a checksummed, previously fitted pipeline.

    The unchanged development loader returns train and validation, filtering
    test rows before materialization. Train is discarded without prediction;
    no fit, calibration, preprocessing update or model/threshold overwrite occurs.
    """
    if not np.isfinite(min_recall) or not 0 <= min_recall <= 1:
        raise ValueError("min_recall must be in [0, 1]")
    for path in (training_metrics, split_report):
        if not path.is_file():
            raise FileNotFoundError(f"Required threshold input not found: {path}")
    recorded = json.loads(training_metrics.read_text(encoding="utf-8"))
    if recorded.get("evaluation_split") != "validation" or recorded.get("test_evaluated") is not False or recorded.get("preprocessing_fit_split") != "train":
        raise ValueError("Saved metrics must describe train-only fitting and validation-only evaluation")
    recorded_model = recorded["models"]["lightgbm"]
    model_path = (training_metrics.parent / recorded_model["artifact"]).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved LightGBM pipeline not found: {model_path}")
    if _sha256(model_path) != recorded_model["sha256"]:
        raise ValueError("Saved LightGBM checksum mismatch")
    if _sha256(split_report) != recorded["data_provenance"]["split_audit_sha256"]:
        raise ValueError("Split audit differs from the model's frozen split audit")
    print("Loading frozen development windows; only validation will be scored...", flush=True)
    development = load_development_data(raw_dir, split_report)
    validation = development.validation
    loader_provenance = development.provenance
    del development  # Training DataFrame is not used by threshold selection.
    if len(validation) != recorded["validation"]["rows"] or int((validation[TARGET_COLUMN] == 1).sum()) != recorded["validation"]["fraud_rows"]:
        raise ValueError("Validation population differs from saved training metrics")
    # Only load the user-owned artifact after verifying its recorded checksum.
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline) or not isinstance(pipeline.named_steps.get("classifier"), lgb.LGBMClassifier):
        raise ValueError("Expected a saved LightGBM sklearn pipeline")
    if not np.array_equal(pipeline.classes_, [0, 1]):
        raise ValueError("Saved pipeline classes must be [0, 1]")
    preprocessor_digest = joblib.hash(pipeline.named_steps["preprocess"])
    labels = validation[TARGET_COLUMN].to_numpy()
    features = validation.drop(columns=[JOIN_KEY, TIME_COLUMN, TARGET_COLUMN])
    print(f"Predicting {len(validation):,} validation rows with the saved pipeline...", flush=True)
    probabilities = pipeline.predict_proba(features)[:, 1]
    if joblib.hash(pipeline.named_steps["preprocess"]) != preprocessor_digest:
        raise ValueError("Prediction changed the fitted preprocessor")
    baseline = binary_metrics(labels, probabilities, threshold=0.5)
    _verify_baseline(baseline, recorded_model)
    sweep = threshold_sweep(labels, probabilities)
    comparison = select_operating_points(sweep, min_recall=min_recall)
    if _sha256(model_path) != recorded_model["sha256"]:
        raise ValueError("Saved model changed during threshold optimization")
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_run_id": recorded["run_id"], "model_sha256": recorded_model["sha256"],
        "training_metrics_sha256": _sha256(training_metrics),
        "split_audit_sha256": loader_provenance["split_audit_sha256"],
        "evaluation_split": "validation", "selection_split": "validation",
        "validation_rows": len(validation), "fraud_rows": int((labels == 1).sum()),
        "min_recall": min_recall, "candidate_thresholds": len(sweep),
        "prediction_rule": "score >= threshold; tied scores move together",
        "candidate_policy": "all distinct validation scores plus 0, 0.5, 1, and nextafter(max_score,+inf) for predict-none",
        "tie_breaking": "higher recall, then higher precision (max F1), then higher threshold",
        "confusion_matrix_layout": "[[TN, FP], [FN, TP]]; actual rows, predicted columns",
        "baseline_reproduced": True, "model_file_unchanged": True, "preprocessor_unchanged": True,
        "retrained": False, "test_materialized": False, "test_evaluated": False,
        "training_defaults_changed": False, "raw_transactions_exported": False,
        "loader": "existing load_development_data; shared CSV scans filter out test rows before materialization; only validation predicted",
        "validation_ap": baseline["pr_auc"], "validation_roc_auc": baseline["roc_auc"],
        "operating_points": [
            {**row, "confusion_matrix": [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]}
            for row in comparison.to_dict(orient="records")
        ],
    }
    write_threshold_reports(sweep, comparison, metadata, output_dir)
    return comparison


def write_threshold_reports(sweep: pd.DataFrame, comparison: pd.DataFrame, metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_threshold_charts(sweep, comparison, output_dir, min_recall=metadata["min_recall"])
    sweep.to_csv(output_dir / "threshold_metrics.csv", index=False, float_format="%.17g")
    comparison.to_csv(output_dir / "threshold_comparison.csv", index=False, float_format="%.17g")
    (output_dir / "threshold_optimization.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Validation-only LightGBM threshold optimization", "",
        f"Saved model run `{metadata['model_run_id']}`. {metadata['validation_rows']:,} validation transactions; "
        f"{metadata['fraud_rows']:,} fraud labels. No retraining, test evaluation, or training-default changes.", "",
        "| Policy | Threshold | Precision | Recall | F1 | TN | FP | FN | TP | Flagged % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.to_dict(orient="records"):
        lines.append(f"| {row['policy']} | {row['threshold']:.10f} | {row['precision']:.6f} "
                     f"| {row['recall']:.6f} | {row['f1']:.6f} | {row['tn']:,} | {row['fp']:,} "
                     f"| {row['fn']:,} | {row['tp']:,} | {row['flagged_pct']:.3f} |")
    baseline, best_f1, constrained = (comparison.iloc[index] for index in range(3))
    lines.extend([
        "", f"- Max-F1 change versus 0.5: F1 {best_f1.f1 - baseline.f1:+.6f}, "
        f"precision {best_f1.precision - baseline.precision:+.6f}, recall {best_f1.recall - baseline.recall:+.6f}.",
        f"- Max-precision with recall >= {metadata['min_recall']:.2f}: precision change "
        f"{constrained.precision - baseline.precision:+.6f}, recall {constrained.recall:.6f} "
        f"({int(constrained.tp):,}/{metadata['fraud_rows']:,} fraud transactions detected).",
        "- No operating point is applied automatically. Choose based on review capacity and error costs; "
        "max F1 and recall-constrained precision are different objectives.", "",
        "## Charts", "",
        "![Precision, recall and F1 over thresholds](threshold_metrics.png)", "",
        "![Precision-recall operating points](threshold_precision_recall.png)", "",
        "![Confusion matrices](threshold_confusion_matrices.png)", "",
        "## Method and integrity", "",
        f"- Evaluated {metadata['candidate_thresholds']:,} exact candidate thresholds: every distinct validation "
        "score plus 0, 0.5, 1 and one representable value above the maximum score (predict-none). "
        "The last candidate can exceed 1 by one floating-point step. No coarse grid, interpolation or rounded-threshold selection.",
        "- Predictions use `score >= threshold`. Equal scores enter/leave together. No-alert precision is "
        "defined as zero. Confusion matrices are `[[TN, FP], [FN, TP]]`, with actual rows and predicted columns.",
        f"- Optimize F1 without a recall constraint, and precision subject to recall >= {metadata['min_recall']:.2f}. "
        "Ties prefer higher recall, then higher precision for max F1, then higher threshold. The recall floor is not relaxed.",
        "- Existing chronological split and loader reused unchanged. The loader returns train/validation after filtering "
        "test rows from shared CSV scans; train is discarded and only validation is scored. No test DataFrame is materialized.",
        "- Saved model checksum and split-audit fingerprint verified; validation metrics at 0.5 reproduce the "
        "saved training run. Prediction leaves the preprocessor and model file unchanged. No fit or calibration calls.",
        f"- Score ranking is unchanged: validation AP={metadata['validation_ap']:.6f}, "
        f"ROC-AUC={metadata['validation_roc_auc']:.6f}. Only the decision threshold changes; outputs remain uncalibrated scores.",
        "- These thresholds are selected and measured on the same validation sample, which was also used for "
        "model selection/early stopping. Improvements are development estimates, not unbiased test performance "
        "or a guarantee of future recall. The test holdout remains reserved.",
        "- Only aggregate metrics, thresholds and charts are exported, never IDs, features or per-transaction predictions. "
        "Full-precision thresholds are retained in JSON/CSV; displayed values are rounded and must not be used for exact replay.", "",
        "[All threshold metrics](threshold_metrics.csv) | [Operating points](threshold_comparison.csv) | "
        "[Exact thresholds and provenance](threshold_optimization.json)",
    ])
    (output_dir / "threshold_optimization.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-report", type=Path, default=Path("reports/chronological_split_summary.json"))
    parser.add_argument("--training-metrics", type=Path, default=Path("reports/training_metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--min-recall", type=float, default=0.70)
    args = parser.parse_args(argv)
    try:
        comparison = optimize_saved_thresholds(args.raw_dir, args.split_report, args.training_metrics, args.output_dir,
                                               min_recall=args.min_recall)
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error, lgb.basic.LightGBMError) as exc:
        print(f"Threshold optimization failed: {exc}", file=sys.stderr)
        return 2
    print(comparison[["policy", "threshold", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]].to_string(index=False))
    print(f"Aggregate reports/charts saved to {args.output_dir}; model and training behavior unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
