"""One-time final test evaluation at the already frozen validation threshold.

Re-running the command reads cached aggregate results; it never rescores test.
No training, threshold search, calibration, or transaction export is performed.
"""

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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline

from .data import DataPaths, JOIN_KEY, TARGET_COLUMN, TIME_COLUMN, merge_transaction_identity
from .features import NUMERIC_FEATURES
from .training import binary_metrics
from .training_data import _header, _quoted

METRICS = ("pr_auc", "roc_auc", "precision", "recall", "f1")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_inputs(reports: Path) -> tuple[dict, dict, dict, dict, Path, dict]:
    training_path = reports / "training_metrics.json"
    split_path = reports / "chronological_split_summary.json"
    threshold_path = reports / "threshold_optimization.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    model = training["models"]["lightgbm"]
    model_path = (reports / model["artifact"]).resolve()
    fingerprints = {
        "model_sha256": _hash(model_path), "training_metrics_sha256": _hash(training_path),
        "split_audit_sha256": _hash(split_path), "threshold_audit_sha256": _hash(threshold_path),
    }
    if (
        training.get("preprocessing_fit_split") != "train"
        or training.get("evaluation_split") != "validation" or training.get("test_evaluated") is not False
        or threshold.get("selection_split") != "validation" or threshold.get("evaluation_split") != "validation"
        or threshold.get("test_evaluated") is not False
        or threshold["model_run_id"] != training["run_id"]
        or fingerprints["model_sha256"] != model["sha256"]
        or fingerprints["model_sha256"] != threshold["model_sha256"]
        or fingerprints["training_metrics_sha256"] != threshold["training_metrics_sha256"]
        or fingerprints["split_audit_sha256"] != threshold["split_audit_sha256"]
        or fingerprints["split_audit_sha256"] != training["data_provenance"]["split_audit_sha256"]
    ):
        raise ValueError("Frozen model, threshold, training and split provenance do not agree")
    for name, expected in training["source_code_sha256"].items():
        if _hash(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Existing training/preprocessing source changed: {name}")
    choices = [point for point in threshold["operating_points"] if point["policy"] == "max_precision_at_recall"]
    if len(choices) != 1:
        raise ValueError("Expected exactly one frozen max_precision_at_recall operating point")
    chosen = choices[0]
    if not np.isfinite(chosen["threshold"]) or not 0 <= chosen["threshold"] <= 1 or chosen["recall"] < threshold["min_recall"]:
        raise ValueError("Invalid frozen validation threshold or recall constraint")
    windows = {row["split"]: row for row in split["splits"]}
    if (
        len(split["splits"]) != 3 or set(windows) != {"train", "validation", "test"}
        or split.get("target_percentages") != [70, 15, 15]
        or not split.get("integrity") or any(value is not True for value in split["integrity"].values())
        or sum(row["rows"] for row in windows.values()) != split["source_rows"]
    ):
        raise ValueError("Invalid frozen split integrity report")
    for row in windows.values():
        if not 0 <= row["TransactionDT_min"] <= row["TransactionDT_max"] or row["rows"] < 1:
            raise ValueError("Invalid chronological window")
    if not (windows["train"]["TransactionDT_max"] < windows["validation"]["TransactionDT_min"]
            and windows["validation"]["TransactionDT_max"] < windows["test"]["TransactionDT_min"]):
        raise ValueError("Frozen chronological windows overlap")
    if (threshold["validation_rows"] != windows["validation"]["rows"]
            or threshold["fraud_rows"] != windows["validation"]["fraud_rows"]):
        raise ValueError("Validation comparison population does not match the split")
    return training, split, threshold, chosen, model_path, fingerprints


def _load_test(raw_dir: Path, split: dict, selected: list[str]) -> tuple[pd.DataFrame, dict]:
    """Materialize only test features/labels; audit other IDs/times in SQL only."""
    paths = DataPaths.from_raw_dir(raw_dir)
    for path in (paths.transaction, paths.identity):
        if not path.is_file():
            raise FileNotFoundError(f"Required holdout input not found: {path}")
    if paths.transaction.stat().st_size != split["source_bytes"]:
        raise ValueError("Transaction CSV size differs from the frozen audit")
    window = next(row for row in split["splits"] if row["split"] == "test")
    low, high = window["TransactionDT_min"], window["TransactionDT_max"]
    transaction_header, identity_header = _header(paths.transaction), _header(paths.identity)
    if not {JOIN_KEY, TIME_COLUMN, TARGET_COLUMN}.issubset(transaction_header) or JOIN_KEY not in identity_header:
        raise ValueError("CSV input is missing required key/time/label columns")
    tx_features = [name for name in selected if name in transaction_header]
    id_features = [name for name in selected if name in identity_header]
    if set(tx_features) & set(id_features) or set(tx_features + id_features) != set(selected):
        raise ValueError("Input features do not match the saved pipeline schema")
    projection = [_quoted(JOIN_KEY)]
    for name in (TIME_COLUMN, TARGET_COLUMN, *tx_features):
        quoted = _quoted(name)
        projection.append(f"CAST({quoted} AS DOUBLE) AS {quoted}" if name in NUMERIC_FEATURES or name in {TIME_COLUMN, TARGET_COLUMN} else quoted)
    csv_options = "header=true, delim=',', all_varchar=true, ignore_errors=false"
    with duckdb.connect(config={"threads": 4, "memory_limit": "1GB"}) as connection:
        connection.execute("SET enable_progress_bar = false")
        connection.execute(
            f"CREATE TEMP TABLE holdout AS SELECT {', '.join(projection)} FROM read_csv(?, {csv_options}) "
            "WHERE CAST(TransactionDT AS DOUBLE) BETWEEN ? AND ?",
            [str(paths.transaction), low, high],
        )
        # No train/validation feature or label DataFrame is loaded for this audit.
        overlap = connection.execute(
            f"SELECT COUNT(*) FROM read_csv(?, {csv_options}) AS other "
            "WHERE other.TransactionID IN (SELECT TransactionID FROM holdout) "
            "AND (other.TransactionDT IS NULL OR CAST(other.TransactionDT AS DOUBLE) NOT BETWEEN ? AND ?)",
            [str(paths.transaction), low, high],
        ).fetchone()[0]
        if overlap:
            raise ValueError("Test TransactionID overlaps rows outside the test window")
        transactions = connection.execute("SELECT * FROM holdout ORDER BY TransactionDT").df()
        identity = connection.execute(
            f"SELECT {', '.join(_quoted(name) for name in [JOIN_KEY, *id_features])} "
            f"FROM read_csv(?, {csv_options}) WHERE TransactionID IN (SELECT TransactionID FROM holdout)",
            [str(paths.identity)],
        ).df()
    if transactions[JOIN_KEY].isna().any() or transactions[JOIN_KEY].duplicated().any():
        raise ValueError("Test IDs must be non-null and unique")
    times, labels = transactions[TIME_COLUMN], transactions[TARGET_COLUMN]
    if (len(transactions) != window["rows"] or times.min() != low or times.max() != high
            or not times.is_monotonic_increasing or not np.isfinite(times).all()
            or labels.isna().any() or not labels.isin([0, 1]).all()
            or int((labels == 1).sum()) != window["fraud_rows"] or labels.nunique() != 2):
        raise ValueError("Test rows, labels or time boundaries disagree with the frozen split")
    merged = merge_transaction_identity(transactions, identity)
    for column in (JOIN_KEY, TIME_COLUMN, TARGET_COLUMN):
        if not merged[column].equals(transactions[column]):
            raise ValueError(f"Identity join changed test {column}")
    return merged, {
        "rows": len(merged), "unique_ids": int(merged[JOIN_KEY].nunique()),
        "fraud_rows": int((labels == 1).sum()), "time_min": float(low), "time_max": float(high),
        "outside_window_id_overlap": int(overlap), "identity_matches": len(identity),
        "join_preserved_rows_labels_and_times": True,
        "only_test_features_and_labels_materialized": True,
        "source_check": "frozen file size, window bounds, counts and label totals; no original per-row hash manifest exists",
    }


def _write_reports(result: dict, reports: Path) -> None:
    """Render only cached aggregates, never reload test or recalculate thresholds."""
    rows = []
    for name in ("validation", "test"):
        values = result[name]
        tn, fp = values["confusion_matrix"][0]
        fn, tp = values["confusion_matrix"][1]
        rows.append({"split": name, "rows": values["rows"], "fraud_rows": values["fraud_rows"],
                     "threshold": result["threshold"], **{key: values[key] for key in METRICS},
                     "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    pd.DataFrame(rows).to_csv(reports / "holdout_comparison.csv", index=False, float_format="%.17g")
    figure = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    positions = np.arange(len(METRICS))
    for offset, row, color in zip((-0.18, 0.18), rows, ("#64748b", "#0f766e")):
        bars = axis.bar(positions + offset, [row[key] for key in METRICS], 0.36, label=row["split"].capitalize(), color=color)
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    axis.set(xticks=positions, xticklabels=["PR-AUC (AP)", "ROC-AUC", "Precision", "Recall", "F1"],
             ylim=(0, 1.03), ylabel="Metric", title="Frozen LightGBM | validation vs final test")
    axis.legend()
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.18)
    figure.savefig(reports / "holdout_metrics.png", dpi=160)
    figure = Figure(figsize=(10, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, row in zip(axes, rows):
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        heatmap = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                axis.text(j, i, f"{matrix[i, j]:,}\n({normalized[i, j]:.1%})", ha="center", va="center",
                          color="white" if normalized[i, j] > 0.55 else "#172033", fontsize=11)
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-fraud", "Fraud"], yticklabels=["Non-fraud", "Fraud"],
                 xlabel="Predicted", ylabel="Actual", title=row["split"].capitalize())
    figure.colorbar(heatmap, ax=list(axes), shrink=0.8, label="Fraction of actual class")
    figure.suptitle(f"Frozen threshold {result['threshold']!r} | counts and row percentages")
    figure.savefig(reports / "holdout_confusion_matrices.png", dpi=160)
    lines = [
        "# Final holdout evaluation", "",
        f"One test scoring pass, completed {result['completed_at_utc']}. Model run `{result['model_run_id']}`. "
        f"Frozen `max_precision_at_recall` threshold: **{result['threshold']!r}**; predict fraud when score >= threshold.", "",
        "| Split | Rows | Fraud rows | PR-AUC (AP) | ROC-AUC | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['split']} | {row['rows']:,} | {row['fraud_rows']:,} | " + " | ".join(f"{row[key]:.6f}" for key in METRICS) + " |")
    lines.extend(["", "Confusion matrices `[[TN, FP], [FN, TP]]` (actual rows, predicted columns):", ""])
    for row in rows:
        lines.append(f"- {row['split']}: `[[{row['tn']}, {row['fp']}], [{row['fn']}, {row['tp']}]]`.")
    lines.extend([
        "", "Test minus validation: " + ", ".join(f"{key} {result['test'][key] - result['validation'][key]:+.6f}" for key in METRICS) + ".", "",
        f"The validation recall floor was {result['validation_recall_floor']:.0%}; final test recall is "
        f"{result['test']['recall']:.2%}. It is reported as observed, without adjusting the threshold.", "",
        "![Validation vs test metrics](holdout_metrics.png)", "",
        "![Validation vs test confusion matrices](holdout_confusion_matrices.png)", "",
        "## Integrity and scope", "",
        "- Saved model, training metrics, split audit and threshold audit fingerprints checked before test loading. "
        "Existing training/preprocessing source is unchanged. The exact JSON threshold was fixed before scoring.",
        "- Only test-window features/labels and matching identity rows were materialized. SQL scanned other IDs/times "
        "solely to confirm no test-ID overlap; no train/validation feature or label DataFrame was loaded.",
        "- Test IDs unique/non-null; expected count, binary-label total, time bounds and order verified. "
        "Frozen train/validation/test time ranges are strictly separated. Join preserves every test row, label and time. "
        "These checks do not replace an unavailable original per-row raw-data hash manifest.",
        "- One predict_proba call; no retraining, fitting, calibration, threshold sweep or test-based model selection. "
        "Saved validation metrics are reused, not recomputed. No other model (including no-skill) was evaluated.",
        "- PR-AUC is average precision; scores are unchanged and uncalibrated. Validation was used for model/threshold "
        "selection, whereas the final test result was not used for tuning. The holdout is now consumed: do not retune against it.",
        "- Only aggregate metrics/charts exported; no transaction rows, IDs or individual predictions. "
        "Model artifacts, training behavior, README and pytest files remain unchanged.",
        "- Re-running the command renders these cached aggregates without loading or predicting test again. "
        "An interrupted scoring pass is blocked rather than automatically retried.", "",
        "[Aggregate metrics/provenance](holdout_evaluation.json) | [Comparison CSV](holdout_comparison.csv)",
    ])
    (reports / "holdout_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_holdout(raw_dir: Path = Path("data/raw"), reports: Path = Path("reports")) -> dict:
    training, split, threshold_audit, chosen, model_path, fingerprints = _frozen_inputs(reports)
    ledger = reports / "holdout_evaluation.json"
    if ledger.exists():
        result = json.loads(ledger.read_text(encoding="utf-8"))
        if result.get("status") != "complete" or result.get("fingerprints") != fingerprints:
            raise ValueError("Holdout evaluation already started or provenance changed; refusing another scoring pass")
        print("Using completed holdout metrics; no test loading or prediction.", flush=True)
        _write_reports(result, reports)
        return result
    pipeline = joblib.load(model_path)  # Previously created local artifact; checksum verified.
    if not isinstance(pipeline, Pipeline) or not isinstance(pipeline.named_steps.get("classifier"), lgb.LGBMClassifier):
        raise ValueError("Expected the saved LightGBM pipeline")
    if not np.array_equal(pipeline.classes_, [0, 1]):
        raise ValueError("Unexpected saved model class order")
    selected = pipeline.named_steps["preprocess"].named_steps["policy"].selected_columns_
    print(f"Loading test window only; threshold frozen at {chosen['threshold']!r}...", flush=True)
    test, integrity = _load_test(raw_dir, split, selected)
    before = joblib.hash(pipeline.named_steps["preprocess"])
    result = {
        "status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprints": fingerprints, "model_run_id": training["run_id"],
        "threshold_policy": "max_precision_at_recall", "threshold": chosen["threshold"],
        "validation_recall_floor": threshold_audit["min_recall"], "integrity": integrity,
        "validation": {
            "rows": threshold_audit["validation_rows"], "fraud_rows": threshold_audit["fraud_rows"],
            "pr_auc": threshold_audit["validation_ap"], "roc_auc": threshold_audit["validation_roc_auc"],
            **{key: chosen[key] for key in ("precision", "recall", "f1", "confusion_matrix")},
        },
    }
    # Exclusive creation prevents concurrent or accidental second scoring passes.
    with ledger.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
    print(f"Scoring {len(test):,} holdout rows once...", flush=True)
    scores = pipeline.predict_proba(test.drop(columns=[JOIN_KEY, TIME_COLUMN, TARGET_COLUMN]))[:, 1]
    result["test"] = binary_metrics(test[TARGET_COLUMN].to_numpy(), scores, threshold=chosen["threshold"])
    result["preprocessor_unchanged"] = joblib.hash(pipeline.named_steps["preprocess"]) == before
    result["model_file_unchanged"] = _hash(model_path) == fingerprints["model_sha256"]
    if not result["preprocessor_unchanged"] or not result["model_file_unchanged"]:
        raise ValueError("Model/preprocessor changed during holdout inference")
    result.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  scoring_passes=1, retrained=False, threshold_tuned_on_test=False, raw_rows_exported=False)
    # Persist aggregates before rendering so a chart failure never requires rescoring.
    ledger.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_reports(result, reports)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    try:
        result = evaluate_holdout(args.raw_dir, args.reports_dir)
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error, lgb.basic.LightGBMError) as exc:
        print(f"Holdout evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"threshold": result["threshold"], "test": result["test"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
