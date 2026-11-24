"""Grouped TreeSHAP for a deterministic validation sample, never the test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from scipy.special import expit
from sklearn.pipeline import Pipeline

from .data import DataPaths, JOIN_KEY, TIME_COLUMN
from .features import NUMERIC_FEATURES
from .training_data import _header, _quoted


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_validation(raw_dir: Path, split: dict, selected: list[str], size: int, seed: int) -> pd.DataFrame:
    """Only sample feature rows reach Python. Labels are not read or used."""
    paths = DataPaths.from_raw_dir(raw_dir)
    if paths.transaction.stat().st_size != split["source_bytes"]:
        raise ValueError("Raw transaction size differs from the frozen split audit")
    window = next(row for row in split["splits"] if row["split"] == "validation")
    tx_header, id_header = _header(paths.transaction), _header(paths.identity)
    tx_features = [name for name in selected if name in tx_header]
    id_features = [name for name in selected if name in id_header]
    if set(tx_features) & set(id_features) or set(tx_features + id_features) != set(selected):
        raise ValueError("Raw feature schema does not match the saved preprocessor")
    csv = "header=true, delim=',', all_varchar=true, ignore_errors=false"
    with duckdb.connect(config={"threads": 4, "memory_limit": "1GB"}) as connection:
        connection.execute("SET enable_progress_bar = false")
        connection.execute(
            f"CREATE TEMP TABLE validation_keys AS SELECT TransactionID, CAST(TransactionDT AS DOUBLE) AS TransactionDT "
            f"FROM read_csv(?, {csv}) WHERE CAST(TransactionDT AS DOUBLE) BETWEEN ? AND ?",
            [str(paths.transaction), window["TransactionDT_min"], window["TransactionDT_max"]],
        )
        observed = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT TransactionID), COUNT(*) FILTER (WHERE TransactionID IS NULL), "
            "MIN(TransactionDT), MAX(TransactionDT) FROM validation_keys"
        ).fetchone()
        if observed != (window["rows"], window["rows"], 0, window["TransactionDT_min"], window["TransactionDT_max"]):
            raise ValueError("Validation IDs, count or bounds differ from the frozen split")
        # Hash ordering is stable even if CSV row order changes. No label/score selection.
        connection.execute(
            "CREATE TEMP TABLE sample_ids AS SELECT TransactionID, md5(TransactionID || ':' || ?) AS sample_key "
            "FROM validation_keys ORDER BY sample_key, TransactionID LIMIT ?", [str(seed), size],
        )
        projection = ["t.TransactionID", "CAST(t.TransactionDT AS DOUBLE) AS TransactionDT"]
        for name in tx_features:
            column = _quoted(name)
            projection.append(f"CAST(t.{column} AS DOUBLE) AS {column}" if name in NUMERIC_FEATURES else f"t.{column}")
        transactions = connection.execute(
            f"SELECT {', '.join(projection)} FROM read_csv(?, {csv}) t JOIN sample_ids s USING (TransactionID) "
            "WHERE CAST(t.TransactionDT AS DOUBLE) BETWEEN ? AND ? ORDER BY s.sample_key, t.TransactionID",
            [str(paths.transaction), window["TransactionDT_min"], window["TransactionDT_max"]],
        ).df()
        identity = connection.execute(
            f"SELECT {', '.join(_quoted(name) for name in [JOIN_KEY, *id_features])} FROM read_csv(?, {csv}) "
            "WHERE TransactionID IN (SELECT TransactionID FROM sample_ids)", [str(paths.identity)],
        ).df()
    sample = transactions.merge(identity, on=JOIN_KEY, how="left", sort=False, validate="one_to_one")
    if len(sample) != min(size, window["rows"]) or not sample[JOIN_KEY].equals(transactions[JOIN_KEY]):
        raise ValueError("Sample membership/order changed during the identity join")
    return sample


def encoded_groups(preprocessor, width: int) -> tuple[list[str], np.ndarray]:
    """Map every fitted output column, including missing indicators/rare buckets.

    Use fitted transformer metadata rather than splitting feature/category names
    on underscores, which would break fields such as P_emaildomain and id_30.
    """
    original = preprocessor.named_steps["policy"].selected_columns_
    columns = preprocessor.named_steps["columns"]
    groups = np.full(width, None, dtype=object)
    for name, transformer, inputs in columns.transformers_:
        section = columns.output_indices_[name]
        if section.start == section.stop:
            continue
        if name == "numeric":
            names = transformer.named_steps["impute"].get_feature_names_out(inputs)
            mapped = [value if value in inputs else value.removeprefix("missingindicator_") for value in names]
        elif name == "categorical":
            counts = [len(values) for values in transformer.categories_]
            rare = getattr(transformer, "infrequent_categories_", [None] * len(counts))
            for index, values in enumerate(rare):
                if values is not None:
                    counts[index] -= len(values) - 1
                if transformer.drop_idx_ is not None and transformer.drop_idx_[index] is not None:
                    counts[index] -= 1
            mapped = np.repeat(inputs, counts).tolist()
        else:
            raise ValueError(f"Unsupported fitted transformer: {name}")
        if len(mapped) != section.stop - section.start:
            raise ValueError(f"Encoded-column mapping mismatch: {name}")
        groups[section] = mapped
    if any(value not in original for value in groups):
        raise ValueError("An encoded SHAP column was not assigned to an original feature")
    return original, groups


def explain_saved(raw_dir: Path, reports: Path, *, sample_size: int = 2000, seed: int = 42) -> dict:
    if not 1 <= sample_size <= 2000 or not 0 <= seed < 2**32:
        raise ValueError("sample_size must be 1..2000; seed must be 0..2**32-1")
    metrics_path, split_path = reports / "training_metrics.json", reports / "chronological_split_summary.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    record = metrics["models"]["lightgbm"]
    model_path = (reports / record["artifact"]).resolve()
    if (metrics.get("preprocessing_fit_split") != "train" or metrics.get("evaluation_split") != "validation"
            or _hash(model_path) != record["sha256"]
            or _hash(split_path) != metrics["data_provenance"]["split_audit_sha256"]):
        raise ValueError("Saved model/split provenance does not match the training run")
    for name, digest in metrics["source_code_sha256"].items():
        if _hash(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"Existing training/preprocessing source changed: {name}")
    pipeline = joblib.load(model_path)  # Trusted local model; fingerprint checked above.
    if not isinstance(pipeline, Pipeline) or not isinstance(pipeline.named_steps.get("classifier"), lgb.LGBMClassifier):
        raise ValueError("Expected the saved LightGBM pipeline")
    if not np.array_equal(pipeline.classes_, [0, 1]):
        raise ValueError("Unexpected model classes")
    preprocess, classifier = pipeline.named_steps["preprocess"], pipeline.named_steps["classifier"]
    before = joblib.hash(preprocess)
    print(f"Loading at most {sample_size:,} validation rows, hash seed {seed}...", flush=True)
    sample = _sample_validation(raw_dir, split, preprocess.named_steps["policy"].selected_columns_, sample_size, seed)
    encoded = preprocess.transform(sample.drop(columns=[JOIN_KEY, TIME_COLUMN]))
    original, mapping = encoded_groups(preprocess, encoded.shape[1])
    iterations = classifier.best_iteration_ or None
    print(f"Computing TreeSHAP for {len(sample):,} rows and {encoded.shape[1]} encoded features...", flush=True)
    # Sparse TreeSHAP returns the input dtype. Promote the same feature values
    # to float64 to avoid float32 attribution-summation roundoff; do not refit.
    contributions = classifier.booster_.predict(encoded.astype(np.float64), pred_contrib=True, num_iteration=iterations, num_threads=4)
    contributions = contributions.toarray() if sparse.issparse(contributions) else np.asarray(contributions)
    if contributions.shape != (len(sample), encoded.shape[1] + 1) or not np.isfinite(contributions).all():
        raise ValueError("Unexpected binary TreeSHAP output")
    values, base = contributions[:, :-1], contributions[:, -1]
    grouped = np.column_stack([values[:, mapping == name].sum(axis=1) for name in original])
    raw = classifier.booster_.predict(encoded, raw_score=True, num_iteration=iterations, num_threads=4)
    probabilities = classifier.predict_proba(encoded, num_iteration=iterations, num_threads=4)[:, 1]
    error = float(np.max(np.abs(base + grouped.sum(axis=1) - raw)))
    if not np.allclose(base + grouped.sum(axis=1), raw, rtol=1e-7, atol=1e-7) or not np.allclose(expit(raw), probabilities, rtol=1e-7, atol=1e-7):
        raise ValueError("Grouped SHAP values do not reconstruct saved model predictions")
    if joblib.hash(preprocess) != before or _hash(model_path) != record["sha256"]:
        raise ValueError("Saved model/preprocessing changed during explanation")
    importance = pd.DataFrame({"feature": original, "mean_abs_grouped_shap": np.abs(grouped).mean(axis=0),
                               "encoded_columns": [int((mapping == name).sum()) for name in original]})
    importance = importance.sort_values(["mean_abs_grouped_shap", "feature"], ascending=[False, True], ignore_index=True)
    reports.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8), layout="constrained")
    ordered = importance.iloc[::-1]
    axis.barh(ordered.feature, ordered.mean_abs_grouped_shap, color="#0f766e")
    axis.set(xlabel="Mean absolute grouped SHAP value (raw log-odds)",
             title=f"Validation sample: {len(sample):,} rows | original-feature TreeSHAP importance")
    axis.grid(axis="x", alpha=0.15)
    axis.set_axisbelow(True)
    figure.savefig(reports / "shap_global_importance.png", dpi=160)
    plt.close(figure)
    explanation = shap.Explanation(values=grouped, base_values=base, feature_names=original)
    # One color: categorical values have no honest low/high numeric ordering.
    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        axis = shap.plots.beeswarm(explanation, max_display=len(original), color="#2563eb", color_bar=False,
                                  plot_size=(10, 9), show=False)
        axis.set(xlabel="Grouped SHAP value (raw log-odds; positive increases fraud score)",
                 title="Validation-only SHAP beeswarm | one dot per sampled row")
        axis.figure.savefig(reports / "shap_beeswarm.png", dpi=160, bbox_inches="tight")
        plt.close(axis.figure)
    finally:
        np.random.set_state(random_state)
    # First hash-ranked case, selected before scores/labels are considered.
    axis = shap.plots.waterfall(explanation[0], max_display=12, show=False)
    axis.set_title(f"Reproducible validation case 0 | model score = {probabilities[0]:.4f}\nContributions in raw log-odds, not probability points", pad=45)
    axis.figure.savefig(reports / "shap_local_prediction.png", dpi=160, bbox_inches="tight")
    plt.close(axis.figure)
    importance.to_csv(reports / "shap_global_importance.csv", index=False)
    result = {
        "model_run_id": metrics["run_id"], "model_sha256": record["sha256"],
        "split_audit_sha256": _hash(split_path), "validation_only": True, "sample_rows": len(sample), "seed": seed,
        "sampling": "ORDER BY md5(TransactionID || ':' || seed), TransactionID within frozen validation window; no labels or scores",
        "sample_ids_sha256": hashlib.sha256("\n".join(sample[JOIN_KEY]).encode()).hexdigest(),
        "encoded_columns": encoded.shape[1], "original_features": len(original), "boosting_iteration": iterations,
        "method": "LightGBM native TreeSHAP; tree-path-dependent training leaf counts, no new background fitted",
        "units": "raw margin/log-odds; sigmoid reconstructs the uncalibrated model score",
        "precision": "float64 native TreeSHAP on unchanged float32-preprocessed feature values",
        "grouping": "sum signed encoded SHAP values per row (including missing indicators), then mean absolute grouped contribution; not recomputed coalition Shapley values",
        "beeswarm_color": "single color avoids imposing an order on nominal categories",
        "max_additivity_error": error, "preprocessor_unchanged": True, "model_file_unchanged": True,
        "retrained": False, "test_used": False, "labels_used": False,
        "local": {"sample_position": 0, "base_value": float(base[0]), "raw_margin": float(raw[0]),
                  "model_score": float(probabilities[0]), "grouped_contributions": dict(zip(original, grouped[0].tolist()))},
        "interpretation": "Model associations on a validation sample, not causal effects or a full-population performance estimate. No transaction IDs, feature values or row-level SHAP matrix exported.",
        "versions": {"shap": shap.__version__, "lightgbm": lgb.__version__, "numpy": np.__version__},
    }
    (reports / "shap_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(importance.head(10).to_string(index=False))
    print(f"Saved three SHAP plots; max additivity error {error:.3g}. No fitting or test data used.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        explain_saved(args.raw_dir, args.reports_dir, sample_size=args.sample_size, seed=args.seed)
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error, lgb.basic.LightGBMError) as exc:
        print(f"SHAP explanation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
