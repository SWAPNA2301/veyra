"""Train and compare local models without loading or scoring the test holdout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .data import JOIN_KEY, TARGET_COLUMN, TIME_COLUMN
from .features import build_preprocessor
from .training_data import load_development_data, validate_development_data


@dataclass(frozen=True)
class TrainingConfig:
    random_state: int = 42
    threads: int = 4
    max_categories: int = 64
    min_frequency: int = 20
    logistic_max_iter: int = 1000
    boosting_rounds: int = 400
    early_stopping_rounds: int = 30
    min_child_samples: int = 100

    def validate(self) -> None:
        if any(getattr(self, key) < 1 for key in (
            "threads", "min_frequency", "logistic_max_iter", "boosting_rounds",
            "early_stopping_rounds", "min_child_samples",
        )) or self.max_categories < 2:
            raise ValueError("Training counts must be positive and max_categories >= 2")


@dataclass
class TrainingResult:
    pipelines: dict[str, Pipeline]
    metrics: dict


def binary_metrics(labels, probabilities, *, threshold: float = 0.5) -> dict:
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("Labels/probabilities must be matching nonempty 1-D arrays")
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Both binary classes are required for ranking metrics")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    predicted = (probabilities >= threshold).astype(np.int8)
    return {
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "threshold": threshold,
        "rows": len(labels),
        "fraud_rows": int((labels == 1).sum()),
    }


def train_models(
    train: pd.DataFrame, validation: pd.DataFrame,
    *, config: TrainingConfig | None = None,
) -> TrainingResult:
    """Fit on train only; validation is transform/evaluation/early-stopping only.

    Deliberately has no test argument. No undersampling, target encoding,
    resampling, threshold search, calibration or train+validation refit occurs.
    """
    config = config or TrainingConfig()
    config.validate()
    validate_development_data(train, validation)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    y_validation = validation[TARGET_COLUMN].to_numpy(dtype=np.int8)
    X_train = train.drop(columns=[JOIN_KEY, TIME_COLUMN, TARGET_COLUMN])
    X_validation = validation.drop(columns=[JOIN_KEY, TIME_COLUMN, TARGET_COLUMN])
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    preprocessor = build_preprocessor(max_categories=config.max_categories, min_frequency=config.min_frequency)
    print(f"Fitting preprocessing on {len(train):,} training rows only...", flush=True)
    train_matrix = preprocessor.fit_transform(X_train)
    validation_matrix = preprocessor.transform(X_validation)
    if not np.isfinite(train_matrix.data).all() or not np.isfinite(validation_matrix.data).all():
        raise ValueError("Preprocessing produced non-finite values")
    print(f"Sparse features: {train_matrix.shape[1]:,}; fitting classifiers...", flush=True)
    models = {
        "no_skill": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced", solver="lbfgs", C=1.0,
            max_iter=config.logistic_max_iter, tol=1e-4, random_state=config.random_state,
        ),
        "lightgbm": lgb.LGBMClassifier(
            objective="binary", metric="average_precision",
            n_estimators=config.boosting_rounds, learning_rate=0.05,
            num_leaves=31, min_child_samples=config.min_child_samples,
            reg_lambda=1.0, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
            scale_pos_weight=negatives / positives,
            random_state=config.random_state, n_jobs=config.threads,
            deterministic=True, force_col_wise=True, verbosity=-1,
        ),
    }
    pipelines, scores = {}, {}
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        started = time.perf_counter()
        with threadpool_limits(limits=config.threads), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if name == "lightgbm":
                # Validation is already transformed by frozen TRAIN statistics.
                model.fit(
                    train_matrix, y_train,
                    eval_set=[(validation_matrix, y_validation)], eval_names=["validation"],
                    callbacks=[lgb.early_stopping(config.early_stopping_rounds, first_metric_only=True, verbose=False)],
                )
            else:
                model.fit(train_matrix, y_train)
        convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
        probabilities = model.predict_proba(validation_matrix)[:, 1]
        scores[name] = binary_metrics(y_validation, probabilities)
        scores[name].update({
            "fit_seconds": time.perf_counter() - started,
            "convergence_warnings": convergence_warnings,
            "iterations": int(model.best_iteration_) if name == "lightgbm" else int(model.n_iter_[0]) if name == "logistic_regression" else 0,
        })
        # Assemble already-fitted components; do NOT call Pipeline.fit on validation.
        pipelines[name] = Pipeline([
            ("preprocess", copy.deepcopy(preprocessor)), ("classifier", model),
        ])
        print(f"{name}: validation AP={scores[name]['pr_auc']:.6f}, ROC-AUC={scores[name]['roc_auc']:.6f}", flush=True)
    winner = max(("logistic_regression", "lightgbm"), key=lambda name: scores[name]["pr_auc"])
    policy = preprocessor.named_steps["policy"]
    result = {
        "evaluation_split": "validation",
        "test_evaluated": False,
        "preprocessing_fit_split": "train",
        "pr_auc_definition": "average precision (step-wise PR integral), not trapezoidal interpolation",
        "threshold_policy": "fixed 0.5 for all models; not tuned",
        "selected_model": winner,
        "selection_rule": "highest validation average precision among the two trained classifiers",
        "refit_train_plus_validation": False,
        "train": {"rows": len(train), "fraud_rows": positives, "fraud_rate": positives / len(train)},
        "validation": {"rows": len(validation), "fraud_rows": int(y_validation.sum()), "fraud_rate": float(y_validation.mean())},
        "class_balance": {
            "logistic_regression": {"0": len(train) / (2 * negatives), "1": len(train) / (2 * positives)},
            "lightgbm_scale_pos_weight": negatives / positives,
            "computed_from": "training labels only; validation metrics are unweighted",
        },
        "features": {
            "policy": "conservative static allowlist, not learned from full-data EDA",
            "selected": policy.selected_columns_,
            "excluded_input_columns": policy.excluded_columns_,
            "train_cardinalities": policy.cardinalities_,
            "preprocessor_fit_rows": policy.fit_rows_,
            "encoded_columns": train_matrix.shape[1],
            "training_matrix_format": "float32 CSR",
            "training_matrix_bytes": int(train_matrix.data.nbytes + train_matrix.indices.nbytes + train_matrix.indptr.nbytes),
        },
        "config": asdict(config),
        "models": scores,
        "versions": {
            "python": platform.python_version(), "pandas": pd.__version__, "duckdb": duckdb.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "lightgbm": lgb.__version__, "joblib": joblib.__version__,
        },
        "model_parameters": {name: model.get_params() for name, model in models.items()},
    }
    for score in scores.values():
        score["pr_auc_lift_over_no_skill"] = score["pr_auc"] / scores["no_skill"]["pr_auc"]
    return TrainingResult(pipelines, result)


def save_training_result(
    result: TrainingResult, *, model_dir: Path, output_dir: Path,
    validation_sample: pd.DataFrame,
) -> None:
    """Save complete pipelines and verify their probability round trip locally."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = model_dir / run_id
    destination.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, pipeline in result.pipelines.items():
        path = destination / f"{name}.joblib"
        joblib.dump(pipeline, path, compress=3)
        restored = joblib.load(path)  # Only artifacts just created by this process.
        expected = pipeline.predict_proba(validation_sample)
        actual = restored.predict_proba(validation_sample)
        if not np.allclose(expected, actual, rtol=0, atol=1e-12):
            raise ValueError(f"Saved pipeline prediction round-trip failed: {name}")
        result.metrics["models"][name].update({
            "artifact": Path(os.path.relpath(path, output_dir)).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "serialization_verified": True,
        })
    q = result.metrics
    q["run_id"] = run_id
    q["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    q["source_code_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("features.py", "training_data.py", "training.py")
    }
    rows = []
    for name, score in q["models"].items():
        tn, fp = score["confusion_matrix"][0]
        fn, tp = score["confusion_matrix"][1]
        rows.append({"model": name, **{key: score[key] for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "threshold")}, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    pd.DataFrame(rows).to_csv(output_dir / "training_metrics.csv", index=False)
    encoded = json.dumps(q, indent=2, allow_nan=False) + "\n"
    (destination / "metrics.json").write_text(encoded, encoding="utf-8")
    (output_dir / "training_metrics.json").write_text(encoded, encoding="utf-8")
    lines = [
        "# Train-only ML pipelines: validation comparison", "",
        f"Run `{run_id}`. Training: {q['train']['rows']:,} rows; validation: {q['validation']['rows']:,} rows. "
        "Existing chronological windows reused unchanged. **Test holdout not evaluated or used for selection.**", "",
        "| Model | PR-AUC (AP) | ROC-AUC | Precision | Recall | F1 | TN | FP | FN | TP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['pr_auc']:.6f} | {row['roc_auc']:.6f} | {row['precision']:.6f} "
                     f"| {row['recall']:.6f} | {row['f1']:.6f} | {row['tn']:,} | {row['fp']:,} | {row['fn']:,} | {row['tp']:,} |")
    lines.extend([
        "", "Confusion matrices use actual rows [non-fraud, fraud] and predicted columns [non-fraud, fraud]: "
        "`[[TN, FP], [FN, TP]]`. Threshold is fixed at 0.5 for every model; no threshold tuning. "
        "Metrics use the original, unweighted validation population.", "",
        f"Selected for future work by validation AP: **{q['selected_model']}**; "
        f"AP lift over no-skill: {q['models'][q['selected_model']]['pr_auc_lift_over_no_skill']:.2f}x. "
        "This is a development comparison, not a held-out test estimate or deployment approval.", "",
        "## Leakage controls and feature policy", "",
        "- Frozen split audit bounds, row counts, label counts and time ranges are checked for train/validation. "
        "IDs are unique and disjoint, with strict chronological boundaries. The API accepts no test DataFrame.",
        "- Shared CSVs must be scanned, but static feature projections and time filters are applied before "
        "materialization; no test features/labels are profiled, transformed, fitted, predicted or scored. "
        "CSV type inference is disabled. No full-data preprocessing or train+validation refit occurs.",
        "- Excluded: TransactionID, isFraud, absolute TransactionDT, identifier-like card/address codes, "
        "anonymous C/D/V aggregates/deltas, and all fields outside the allowlist. These anonymous features "
        "are withheld because their past-only availability is unverified, not because leakage was proven.",
        "- Included fields: " + ", ".join(q["features"]["selected"]) + ". "
        "id_30/id_31/id_33 are device OS/browser/screen attributes, not row IDs. "
        "This conservative baseline deliberately does not use every competition feature.",
        "- Train-only median imputation (all-empty numeric fields retained with zero fallback), missing indicators "
        "and scaling; categorical missing tokens and train-fitted rare-category grouping. Unknown categories "
        "use the infrequent bucket when available, otherwise all-zero encoding. No target encoding.",
        f"- One-hot outputs capped at {q['config']['max_categories']} per categorical field; minimum frequency "
        f"{q['config']['min_frequency']}. {q['features']['encoded_columns']} encoded features; "
        f"float32 CSR training matrix: {q['features']['training_matrix_bytes'] / 1024**2:.2f} MiB.",
        "- Logistic Regression uses train-derived balanced class weights. LightGBM uses train negative/positive "
        f"ratio {q['class_balance']['lightgbm_scale_pos_weight']:.6f}; no resampling. "
        "Weighted-model outputs are not calibrated fraud probabilities.",
        f"- LightGBM early stopping monitors validation average precision only; best iteration "
        f"{q['models']['lightgbm']['iterations']}. Logistic Regression iterations: {q['models']['logistic_regression']['iterations']}.",
        "- The constant no-skill model uses the training fraud prior. Its validation AP equals validation "
        "prevalence, ROC-AUC is 0.5, and at threshold 0.5 it predicts no fraud for this imbalanced dataset.",
        "- [Full-data weekly EDA](03_time_trends.sql) is not a training feature: its outcome rates include "
        "future periods. No weekly fraud rate or future-derived aggregate is fed to these models.", "",
        "## Saved pipelines", "",
    ])
    for name, score in q["models"].items():
        lines.append(f"- [{name}]({score['artifact']}): fitted preprocessing + classifier; serialization predictions verified.")
        for warning in score["convergence_warnings"]:
            lines.append(f"  Convergence warning: {warning}")
    lines.extend([
        "", "Model artifacts and learned vocabularies are Git-ignored. Load only trusted joblib files, "
        "using this source package and the library versions recorded in the metrics; no serving/UI is added.", "",
        "[Full metrics/config/provenance](training_metrics.json) | [Metrics CSV](training_metrics.csv)", "",
        "## Metric and API references", "",
        "PR-AUC here is [scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), "
        "not trapezoidal PR interpolation. Category handling follows "
        "[OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html); "
        "validation stopping uses [LightGBM early stopping](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html).",
    ])
    (output_dir / "training_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-report", type=Path, default=Path("reports/chronological_split_summary.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    try:
        print("Loading frozen train/validation windows (test excluded)...", flush=True)
        data = load_development_data(args.raw_dir, args.split_report)
        result = train_models(data.train, data.validation)
        result.metrics["data_provenance"] = data.provenance
        save_training_result(result, model_dir=args.model_dir, output_dir=args.output_dir,
                             validation_sample=data.validation.head(64))
    except (OSError, ValueError, KeyError, duckdb.Error, lgb.basic.LightGBMError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 2
    print(f"Saved pipelines and validation metrics. Selected: {result.metrics['selected_model']}; test not evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
