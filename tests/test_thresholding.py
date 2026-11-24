import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from veyra import thresholding
from veyra.thresholding import main, optimize_saved_thresholds, select_operating_points, threshold_sweep
from veyra.training import binary_metrics


def test_all_candidates_match_brute_force_and_ties_stay_together():
    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.0, 0.5, 0.5, 1.0, 1.0, 0.2])
    original_labels, original_scores = labels.copy(), scores.copy()
    sweep = threshold_sweep(labels, scores)
    assert sweep.threshold.is_unique and sweep.threshold.is_monotonic_increasing
    assert sweep.threshold.iloc[-1] > 1  # valid predict-none sentinel
    for row in sweep.itertuples():
        predicted = scores >= row.threshold
        assert [[row.tn, row.fp], [row.fn, row.tp]] == confusion_matrix(labels, predicted, labels=[0, 1]).tolist()
        assert row.precision == pytest.approx(precision_score(labels, predicted, zero_division=0))
        assert row.recall == pytest.approx(recall_score(labels, predicted, zero_division=0))
        assert row.f1 == pytest.approx(f1_score(labels, predicted, zero_division=0))
        assert row.tn + row.fp + row.fn + row.tp == len(labels)
        assert row.flagged_rows == row.tp + row.fp
    at_half = sweep.loc[sweep.threshold == 0.5].iloc[0]
    assert [at_half.tn, at_half.fp, at_half.fn, at_half.tp] == [1, 2, 1, 2]
    assert sweep.iloc[-1].flagged_rows == sweep.iloc[-1].precision == 0
    np.testing.assert_array_equal(original_labels, labels)
    np.testing.assert_array_equal(original_scores, scores)


def test_complete_sweep_contains_maxima_and_meets_recall_constraint():
    generator = np.random.default_rng(17)
    labels = generator.integers(0, 2, 500)
    scores = np.round(generator.random(500), 2)
    sweep = threshold_sweep(labels, scores)
    comparison = select_operating_points(sweep)
    baseline, best_f1, constrained = (comparison.iloc[index] for index in range(3))
    assert baseline.threshold == 0.5
    assert best_f1.f1 == sweep.f1.max()
    assert constrained.recall >= 0.70
    assert constrained.precision == sweep.loc[sweep.recall >= 0.70, "precision"].max()
    assert best_f1.f1 >= baseline.f1
    assert (sweep.tp.diff().dropna() <= 0).all() and (sweep.fp.diff().dropna() <= 0).all()
    assert len(sweep) <= len(np.unique(scores)) + 4


def test_tie_breaks_prefer_recall_then_higher_threshold():
    # F1 ties: detecting the first positive alone vs all rows; recall decides.
    sweep = threshold_sweep([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6])
    selected = select_operating_points(sweep)
    assert selected.iloc[1].recall == 1
    assert selected.iloc[1].threshold == 0.6  # all-positive equivalent cutoffs -> highest
    # Constant scores: threshold 0, 0.4 and 0.5 can describe duplicate decisions.
    constant = select_operating_points(threshold_sweep([0, 1], [0.8, 0.8]))
    assert constant.iloc[1].threshold == 0.8
    assert constant.iloc[2].threshold == 0.8


def test_recall_floor_includes_exact_boundary_and_never_relaxes():
    labels = np.array([1] * 10 + [0] * 10)
    scores = np.array([0.9] * 7 + [0.1] * 3 + [0.05] * 10)
    sweep = threshold_sweep(labels, scores)
    exact = select_operating_points(sweep, min_recall=0.70).iloc[2]
    # Both 0.1 and 0.9 have precision 1: higher recall wins, not arbitrary order.
    assert exact.precision == 1 and exact.recall == 1
    only_high = sweep.loc[sweep.threshold >= 0.5]
    assert select_operating_points(only_high, min_recall=0.70).iloc[2].recall == 0.70
    with pytest.raises(ValueError, match="No threshold satisfies"):
        select_operating_points(only_high, min_recall=np.nextafter(0.70, 1.0))


@pytest.mark.parametrize(("labels", "scores"), [
    ([], []), ([0, 1], [0.2]), ([[0], [1]], [[0.2], [0.8]]),
    ([0, 0], [0.1, 0.2]), ([1, 1], [0.1, 0.2]), ([0, 2], [0.1, 0.2]),
    ([0, np.nan], [0.1, 0.2]), ([0, 1], [np.nan, 0.2]),
    ([0, 1], [0.1, np.inf]), ([0, 1], [-0.1, 0.2]), ([0, 1], [0.1, 1.1]),
])
def test_invalid_labels_and_scores_fail_clearly(labels, scores):
    with pytest.raises(ValueError):
        threshold_sweep(labels, scores)


@pytest.mark.parametrize("floor", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_recall_floor_is_rejected(floor):
    with pytest.raises(ValueError, match="min_recall"):
        select_operating_points(threshold_sweep([0, 1], [0.2, 0.8]), min_recall=floor)


class FrozenPreprocessor:
    def __init__(self):
        self.statistics_ = np.array([10.0, 20.0])

    def fit(self, *args, **kwargs):
        pytest.fail("Threshold optimization must never fit preprocessing")


@pytest.fixture
def saved_fixture(tmp_path: Path, monkeypatch):
    raw, reports, models = tmp_path / "raw", tmp_path / "reports", tmp_path / "models"
    raw.mkdir()
    reports.mkdir()
    models.mkdir()
    labels = np.array([int(index % 5 == 0) for index in range(85)])
    # Test values cannot be parsed as labels or amounts. No test split is created.
    pd.DataFrame({
        "TransactionID": [str(index) for index in range(85)] + ["TEST_DO_NOT_USE"] * 15,
        "TransactionDT": range(100),
        "isFraud": list(labels) + ["DO_NOT_PARSE_TEST"] * 15,
        "TransactionAmt": list(range(1, 86)) + ["DO_NOT_TRANSFORM_TEST"] * 15,
    }).to_csv(raw / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": [str(index) for index in range(100)],
                  "DeviceInfo": [f"device-{index}" for index in range(100)]}).to_csv(raw / "train_identity.csv", index=False)
    split = reports / "split.json"
    split.write_text(json.dumps({
        "target_percentages": [70, 15, 15], "integrity": {"passed": True},
        "source_bytes": (raw / "train_transaction.csv").stat().st_size,
        "splits": [
            {"split": "train", "rows": 70, "fraud_rows": 14, "TransactionDT_min": 0, "TransactionDT_max": 69},
            {"split": "validation", "rows": 15, "fraud_rows": 3, "TransactionDT_min": 70, "TransactionDT_max": 84},
            {"split": "test", "fraud_rows": "MUST_NOT_INSPECT_TEST_METADATA"},
        ],
    }))
    artifact = models / "lightgbm.joblib"
    artifact.write_bytes(b"owned saved artifact stub; no fitting in these tests")
    scores = np.linspace(0.01, 0.99, 15)
    model_metrics = binary_metrics(labels[70:85], scores)
    model_metrics.update({"artifact": "../models/lightgbm.joblib", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})
    metrics = reports / "training_metrics.json"
    metrics.write_text(json.dumps({
        "run_id": "test-saved-run", "evaluation_split": "validation", "test_evaluated": False,
        "preprocessing_fit_split": "train", "validation": {"rows": 15, "fraud_rows": 3},
        "models": {"lightgbm": model_metrics},
        "data_provenance": {"split_audit_sha256": hashlib.sha256(split.read_bytes()).hexdigest()},
    }))
    pipeline = Mock(spec=Pipeline)
    classifier = Mock(spec=lgb.LGBMClassifier)
    pipeline.named_steps = {"classifier": classifier, "preprocess": FrozenPreprocessor()}
    pipeline.classes_ = np.array([0, 1])
    pipeline.fit.side_effect = AssertionError("Must not retrain saved pipeline")
    classifier.fit.side_effect = AssertionError("Must not retrain saved classifier")

    def predict(frame):
        assert len(frame) == 15
        assert frame["TransactionAmt"].min() == 71 and frame["TransactionAmt"].max() == 85
        assert not {"TransactionID", "TransactionDT", "isFraud"}.intersection(frame.columns)
        return np.column_stack((1 - scores, scores))

    pipeline.predict_proba.side_effect = predict
    load = Mock(return_value=pipeline)
    monkeypatch.setattr(thresholding.joblib, "load", load)
    return raw, split, metrics, reports, artifact, pipeline, load


def test_saved_pipeline_validation_only_no_retraining_and_aggregate_outputs(saved_fixture):
    raw, split, metrics, reports, artifact, pipeline, load = saved_fixture
    protected = [split, metrics, artifact, *raw.iterdir()]
    before = {path: path.read_bytes() for path in protected}
    comparison = optimize_saved_thresholds(raw, split, metrics, reports)
    pipeline.predict_proba.assert_called_once()
    pipeline.fit.assert_not_called()
    pipeline.named_steps["classifier"].fit.assert_not_called()
    load.assert_called_once_with(artifact.resolve())
    assert {path: path.read_bytes() for path in protected} == before
    metadata = json.loads((reports / "threshold_optimization.json").read_text())
    assert metadata["baseline_reproduced"] and metadata["preprocessor_unchanged"]
    assert not metadata["retrained"] and not metadata["test_materialized"] and not metadata["test_evaluated"]
    assert not metadata["training_defaults_changed"] and not metadata["raw_transactions_exported"]
    assert metadata["validation_rows"] == 15 and metadata["fraud_rows"] == 3
    assert metadata["operating_points"][2]["recall"] >= 0.70
    assert [row["threshold"] for row in metadata["operating_points"]] == comparison["threshold"].tolist()
    for name in ("threshold_metrics.csv", "threshold_comparison.csv"):
        exported = pd.read_csv(reports / name)
        assert not {"TransactionID", "TransactionDT", "isFraud", "DeviceInfo", "TransactionAmt"}.intersection(exported.columns)
    pd.testing.assert_frame_equal(comparison, pd.read_csv(reports / "threshold_comparison.csv"))
    for name in ("threshold_metrics.png", "threshold_precision_recall.png", "threshold_confusion_matrices.png"):
        assert (reports / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("corruption", ["model_checksum", "split_fingerprint", "baseline", "preprocessing_split"])
def test_mismatch_fails_closed_without_new_reports(saved_fixture, corruption):
    raw, split, metrics, reports, artifact, pipeline, load = saved_fixture
    previous = reports / "threshold_optimization.md"
    previous.write_text("previous successful report")
    record = json.loads(metrics.read_text())
    if corruption == "model_checksum":
        artifact.write_bytes(b"unexpected replacement")
    elif corruption == "split_fingerprint":
        split.write_text(split.read_text() + " ")
    elif corruption == "preprocessing_split":
        record["preprocessing_fit_split"] = "full_data"
        metrics.write_text(json.dumps(record))
    else:
        record["models"]["lightgbm"]["confusion_matrix"][0][0] += 1
        metrics.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        optimize_saved_thresholds(raw, split, metrics, reports)
    if corruption != "baseline":
        load.assert_not_called()
    assert previous.read_text() == "previous successful report"
    assert not list(reports.glob("threshold_*.png"))


def test_cli_missing_input_is_graceful(tmp_path, capsys):
    assert main(["--training-metrics", str(tmp_path / "missing.json")]) == 2
    assert "Required threshold input not found" in capsys.readouterr().err
