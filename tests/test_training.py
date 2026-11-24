import copy
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from veyra.features import FeaturePolicy, build_preprocessor
from veyra.splitting import run_split_report
from veyra.training import TrainingConfig, binary_metrics, main, save_training_result, train_models
from veyra.training_data import load_development_data, validate_development_data


def development_frames():
    train = pd.DataFrame({
        "TransactionID": range(120), "TransactionDT": range(120),
        "isFraud": [int(i % 5 == 0) for i in range(120)],
        "TransactionAmt": [None if i % 13 == 0 else float(100 if i % 5 == 0 else 10) for i in range(120)],
        "dist1": [float(i % 3) for i in range(120)], "dist2": np.nan,
        "ProductCD": ["C" if i % 5 == 0 else "W" for i in range(120)],
        "DeviceInfo": [f"device-{i}" for i in range(120)],
        "card1": range(120), "D1": range(120), "V1": range(120),
        "future_fraud_rate": [int(i % 5 == 0) for i in range(120)],
    })
    validation = train.iloc[:30].copy().reset_index(drop=True)
    validation["TransactionID"] += 1000
    validation["TransactionDT"] += 1000
    validation["DeviceInfo"] = "VALIDATION_ONLY"
    validation["dist2"] = 1000000.0
    return train, validation


@pytest.fixture(scope="module")
def trained():
    train, validation = development_frames()
    config = TrainingConfig(max_categories=8, min_frequency=2, logistic_max_iter=200,
                            boosting_rounds=20, early_stopping_rounds=5, min_child_samples=5, threads=1)
    return train_models(train, validation, config=config)


def test_preprocessing_is_train_only_and_sparse():
    train, validation = development_frames()
    original = train.copy(deep=True)
    preprocessor = build_preprocessor(max_categories=8, min_frequency=2)
    encoded = preprocessor.fit_transform(train)
    before = joblib.hash(preprocessor)
    transformed = preprocessor.transform(validation)
    assert before == joblib.hash(preprocessor)
    assert sparse.isspmatrix_csr(encoded) and sparse.isspmatrix_csr(transformed)
    assert encoded.dtype == np.float32
    assert encoded.shape[1] == transformed.shape[1]
    numeric = preprocessor.named_steps["columns"].named_transformers_["numeric"]
    assert numeric.named_steps["impute"].statistics_[0] == train["TransactionAmt"].median()
    assert numeric.named_steps["impute"].statistics_[2] == 0  # entirely empty training field
    assert numeric.named_steps["scale"].mean_[2] == 0  # validation's million never contributes
    encoder = preprocessor.named_steps["columns"].named_transformers_["categorical"]
    assert all("value:VALIDATION_ONLY" not in categories for categories in encoder.categories_)
    assert len(encoder.get_feature_names_out()) <= 2 * 8
    assert np.isfinite(encoded.data).all() and np.isfinite(transformed.data).all()
    pd.testing.assert_frame_equal(train, original)


def test_feature_policy_excludes_target_identifiers_time_and_unknown_leakage():
    train, _ = development_frames()
    policy = FeaturePolicy().fit(train)
    encoded = policy.transform(train)
    assert set(encoded.columns) == {"TransactionAmt", "dist1", "dist2", "ProductCD", "DeviceInfo"}
    for excluded in ("TransactionID", "TransactionDT", "isFraud", "card1", "D1", "V1", "future_fraud_rate"):
        assert excluded in policy.excluded_columns_ and excluded not in encoded
    altered = train.copy()
    altered["isFraud"] = 1 - altered["isFraud"]
    altered["TransactionID"] += 1000000
    altered["future_fraud_rate"] = 999.0
    pd.testing.assert_frame_equal(encoded, policy.transform(altered))
    with pytest.raises(ValueError, match="Missing fitted feature"):
        policy.transform(train.drop(columns="DeviceInfo"))


def test_missing_token_does_not_collide_with_real_category():
    frame = pd.DataFrame({"TransactionAmt": [1.0, 2.0], "DeviceInfo": [None, "missing:"]})
    normalized = FeaturePolicy().fit_transform(frame)
    assert normalized["DeviceInfo"].tolist() == ["missing:", "value:missing:"]


def test_constant_no_skill_and_known_confusion_metrics():
    labels = np.array([0, 0, 0, 1])
    baseline = binary_metrics(labels, np.full(4, 0.2))
    assert baseline["pr_auc"] == 0.25 and baseline["roc_auc"] == 0.5
    assert baseline["precision"] == baseline["recall"] == baseline["f1"] == 0
    assert baseline["confusion_matrix"] == [[3, 0], [1, 0]]
    perfect = binary_metrics(labels, [0.1, 0.2, 0.3, 0.9])
    assert perfect["pr_auc"] == perfect["roc_auc"] == perfect["f1"] == 1
    mixed = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9])
    assert mixed["confusion_matrix"] == [[1, 1], [1, 1]]
    assert mixed["precision"] == mixed["recall"] == mixed["f1"] == 0.5


@pytest.mark.parametrize("scores", [[np.nan, 0.5], [-0.1, 0.5], [0.3, 1.1], [0.3]])
def test_invalid_scores_rejected(scores):
    with pytest.raises(ValueError):
        binary_metrics([0, 1], scores)


def test_models_use_train_weights_and_validation_only(trained):
    train, validation = development_frames()
    assert "test" not in inspect.signature(train_models).parameters
    assert trained.metrics["test_evaluated"] is False
    assert trained.metrics["features"]["preprocessor_fit_rows"] == len(train)
    assert trained.metrics["class_balance"]["lightgbm_scale_pos_weight"] == 4
    lgbm = trained.pipelines["lightgbm"].named_steps["classifier"]
    assert lgbm.get_params()["scale_pos_weight"] == 4
    assert set(lgbm.evals_result_) == {"validation"}
    assert lgbm.best_iteration_ > 0
    prior = trained.pipelines["no_skill"].named_steps["classifier"].class_prior_
    assert prior.tolist() == [0.8, 0.2]
    for name, pipeline in trained.pipelines.items():
        original = pipeline.predict_proba(validation)
        changed = validation.copy()
        changed["isFraud"] = 1 - changed["isFraud"]
        changed["future_fraud_rate"] = 123456
        np.testing.assert_array_equal(original, pipeline.predict_proba(changed))
        score = trained.metrics["models"][name]
        assert score["rows"] == 30 and sum(map(sum, score["confusion_matrix"])) == 30
        assert score["threshold"] == 0.5
        assert score["convergence_warnings"] == []
    assert trained.metrics["models"]["no_skill"]["pr_auc"] == pytest.approx(validation["isFraud"].mean())


def test_serialized_pipelines_and_metrics_round_trip(trained, tmp_path: Path):
    _, validation = development_frames()
    result = copy.deepcopy(trained)
    models, reports = tmp_path / "models", tmp_path / "reports"
    save_training_result(result, model_dir=models, output_dir=reports, validation_sample=validation)
    metrics = json.loads((reports / "training_metrics.json").read_text())
    assert len(list(models.rglob("*.joblib"))) == 3
    assert len(pd.read_csv(reports / "training_metrics.csv")) == 3
    for name, score in metrics["models"].items():
        assert score["serialization_verified"] is True
        restored = joblib.load(reports / score["artifact"])
        np.testing.assert_allclose(restored.predict_proba(validation), trained.pipelines[name].predict_proba(validation), rtol=0, atol=1e-12)
    assert metrics["test_evaluated"] is False


def fixture_files(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    transactions = pd.DataFrame({
        "TransactionID": range(100), "TransactionDT": range(100),
        "isFraud": [int(index % 5 == 0) for index in range(100)],
        "TransactionAmt": np.arange(100, dtype=float) + 1,
        "ProductCD": ["C" if index % 5 == 0 else "W" for index in range(100)],
        "C1": 999, "V1": 999, "card1": range(100),
    })
    transactions.to_csv(raw / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": range(100), "DeviceInfo": [f"device-{index}" for index in range(100)]}).to_csv(raw / "train_identity.csv", index=False)
    reports = tmp_path / "splits"
    run_split_report(raw, reports)
    return raw, reports / "chronological_split_summary.json", transactions


def test_loader_cannot_consume_poisoned_test_features_or_labels(tmp_path: Path):
    raw, report_path, transactions = fixture_files(tmp_path)
    # Change only the held-out region to unparsable labels/features and duplicate IDs.
    # If the ML loader evaluates those values, this call must fail.
    transactions["isFraud"] = transactions["isFraud"].astype(object)
    transactions["TransactionAmt"] = transactions["TransactionAmt"].astype(object)
    transactions.loc[85:, "isFraud"] = "DO_NOT_EVALUATE_TEST"
    transactions.loc[85:, "TransactionAmt"] = "DO_NOT_TRANSFORM_TEST"
    transactions.loc[85:, "TransactionID"] = 999999
    transactions.to_csv(raw / "train_transaction.csv", index=False)
    audit = json.loads(report_path.read_text())
    audit["source_bytes"] = (raw / "train_transaction.csv").stat().st_size
    report_path.write_text(json.dumps(audit))
    loaded = load_development_data(raw, report_path)
    assert len(loaded.train) == 70 and len(loaded.validation) == 15
    assert loaded.validation["TransactionDT"].max() == 84
    assert not loaded.provenance["test_materialized"] and not loaded.provenance["test_evaluated"]
    assert "C1" not in loaded.train and "V1" not in loaded.train and "card1" not in loaded.train
    assert loaded.train["DeviceInfo"].nunique() == 70
    assert all("device-99" not in frame["DeviceInfo"].tolist() for frame in (loaded.train, loaded.validation))


def test_loader_rejects_stale_split_and_duplicate_identity(tmp_path: Path):
    raw, report_path, _ = fixture_files(tmp_path)
    report = json.loads(report_path.read_text())
    report["source_bytes"] += 1
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="size changed"):
        load_development_data(raw, report_path)
    report["source_bytes"] -= 1
    report_path.write_text(json.dumps(report))
    identity = pd.read_csv(raw / "train_identity.csv")
    pd.concat([identity, identity.iloc[[0]]]).to_csv(raw / "train_identity.csv", index=False)
    with pytest.raises(ValueError, match="duplicate TransactionID"):
        load_development_data(raw, report_path)


@pytest.mark.parametrize("failure", ["time", "id_overlap", "label", "single_class", "unordered"])
def test_training_rejects_bad_split_integrity(failure):
    train, validation = development_frames()
    if failure == "time":
        validation["TransactionDT"] -= 1000
    elif failure == "id_overlap":
        validation["TransactionID"] -= 1000
    elif failure == "label":
        validation.loc[0, "isFraud"] = 2
    elif failure == "single_class":
        train["isFraud"] = 0
    else:
        validation = validation.iloc[::-1]
    with pytest.raises(ValueError):
        validate_development_data(train, validation)


def test_missing_training_files_exit_cleanly(tmp_path: Path, capsys):
    assert main(["--raw-dir", str(tmp_path), "--split-report", str(tmp_path / "absent.json")]) == 2
    assert "Required training input not found" in capsys.readouterr().err
