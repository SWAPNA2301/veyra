import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from veyra.splitting import (
    ChronologicalSplits,
    chronological_split,
    main,
    run_split_report,
    summarize_splits,
    validate_splits,
)


def transactions(times: list[int]) -> pd.DataFrame:
    size = len(times)
    return pd.DataFrame({
        "TransactionID": range(1000, 1000 + size),
        "TransactionDT": times,
        "isFraud": [int(index % 5 == 0) for index in range(size)],
        "optional_feature": [None if index % 3 == 0 else float(index) for index in range(size)],
        "category": pd.Categorical(["a" if index % 2 else "b" for index in range(size)]),
    })


@pytest.fixture
def source() -> pd.DataFrame:
    return transactions(list(range(100)))


def test_exact_proportions_sorting_and_complete_row_preservation(source) -> None:
    shuffled = source.sample(frac=1, random_state=7)
    # DataFrame indices need not be unique: TransactionID is the identity.
    shuffled.index = [index // 2 for index in range(len(shuffled))]
    original = shuffled.copy(deep=True)
    splits = chronological_split(shuffled)
    assert [len(part) for _, part in splits.items()] == [70, 15, 15]
    pd.testing.assert_frame_equal(shuffled, original)
    pd.testing.assert_frame_equal(
        pd.concat([part for _, part in splits.items()]),
        original.sort_values("TransactionDT", kind="mergesort"),
    )
    assert splits.train["TransactionDT"].max() < splits.validation["TransactionDT"].min()
    assert splits.validation["TransactionDT"].max() < splits.test["TransactionDT"].min()
    ids = [set(part["TransactionID"]) for _, part in splits.items()]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert set.union(*ids) == set(source["TransactionID"])
    summary = summarize_splits(splits).set_index("split")
    assert summary["fraud_rows"].tolist() == [14, 3, 3]
    assert summary["fraud_rate_pct"].tolist() == [20.0] * 3
    assert summary["actual_pct"].tolist() == [70.0, 15.0, 15.0]
    splits.train.iloc[0, splits.train.columns.get_loc("isFraud")] = 0
    pd.testing.assert_frame_equal(shuffled, original)  # independent returned frames


def test_labels_and_features_do_not_influence_membership(source) -> None:
    first = chronological_split(source)
    changed = source.copy()
    changed["isFraud"] = 1 - changed["isFraud"]
    changed["optional_feature"] = 999999.0
    second = chronological_split(changed)
    shuffled = chronological_split(source.sample(frac=1, random_state=13))
    for name, part in first.items():
        assert part["TransactionID"].tolist() == getattr(second, name)["TransactionID"].tolist()
        assert part["TransactionID"].tolist() == getattr(shuffled, name)["TransactionID"].tolist()


def test_timestamp_groups_never_cross_either_boundary() -> None:
    frame = transactions([time for time, size in enumerate([10] * 6 + [20, 10, 10]) for _ in range(size)])
    first = chronological_split(frame)
    assert [len(part) for _, part in first.items()] == [60, 20, 20]
    second = chronological_split(frame.sample(frac=1, random_state=9))
    timestamps = [set(part["TransactionDT"]) for _, part in first.items()]
    assert not timestamps[0] & timestamps[1] and not timestamps[1] & timestamps[2]
    for name, part in first.items():
        assert set(part["TransactionID"]) == set(getattr(second, name)["TransactionID"])
    # Stable source order is retained within a timestamp group.
    for _, part in second.items():
        assert part["TransactionDT"].is_monotonic_increasing


def test_large_tie_group_reserves_nonempty_holdouts() -> None:
    source = transactions([0] + [1] * 98 + [2])
    splits = chronological_split(source)
    assert [len(part) for _, part in splits.items()] == [1, 98, 1]
    validate_splits(source, splits)


@pytest.mark.parametrize("size", [3, 4, 7, 19, 20, 23, 101])
def test_small_and_nondivisible_input_is_exhaustive(size) -> None:
    frame = transactions(list(range(size)))
    splits = chronological_split(frame)
    assert all(len(part) > 0 for _, part in splits.items())
    assert sum(len(part) for _, part in splits.items()) == size
    if size >= 7:
        first, second = size * 70 // 100, size * 85 // 100
        assert [len(part) for _, part in splits.items()] == [first, second - first, size - second]


@pytest.mark.parametrize("times", [[], [1], [1, 2], [1] * 20, [1] * 10 + [2] * 10])
def test_insufficient_timestamps_fail_clearly(times) -> None:
    with pytest.raises(ValueError, match="empty|three distinct"):
        chronological_split(transactions(times))


@pytest.mark.parametrize("labels", [[0] * 100, [1] * 100, [0] * 70 + [1] * 30])
def test_single_class_periods_are_not_resampled(source, labels) -> None:
    source["isFraud"] = labels
    splits = chronological_split(source)
    assert [len(part) for _, part in splits.items()] == [70, 15, 15]
    assert summarize_splits(splits)["fraud_rows"].sum() == sum(labels)
    validate_splits(source, splits)


@pytest.mark.parametrize(("column", "value"), [
    ("TransactionID", None),
    ("TransactionDT", None), ("TransactionDT", float("inf")),
    ("TransactionDT", float("-inf")), ("TransactionDT", -1),
    ("TransactionDT", "not-a-time"), ("TransactionDT", True),
    ("isFraud", None), ("isFraud", 2), ("isFraud", "1"),
])
def test_invalid_input_fails_without_mutation(source, column, value) -> None:
    if isinstance(value, str) or isinstance(value, bool):
        source[column] = source[column].astype(object)
    elif value is None or isinstance(value, float):
        source[column] = source[column].astype(float)
    source.loc[0, column] = value
    original = source.copy(deep=True)
    with pytest.raises(ValueError, match=column):
        chronological_split(source)
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize("conflicting_label", [False, True])
def test_duplicate_ids_are_rejected_even_when_labels_agree(source, conflicting_label) -> None:
    repeated = source.iloc[[0]].copy()
    if conflicting_label:
        repeated["isFraud"] = 1 - repeated["isFraud"]
    with pytest.raises(ValueError, match="TransactionID must be unique"):
        chronological_split(pd.concat([source, repeated], ignore_index=True))


@pytest.mark.parametrize("column", ["TransactionID", "TransactionDT", "isFraud"])
def test_missing_required_columns_are_rejected(source, column) -> None:
    with pytest.raises(ValueError, match=column):
        chronological_split(source.drop(columns=column))


def test_duplicate_columns_are_rejected(source) -> None:
    with pytest.raises(ValueError, match="duplicate_columns"):
        chronological_split(pd.concat([source, source[["isFraud"]]], axis=1))


def test_validator_rejects_cross_split_id_overlap(source) -> None:
    splits = chronological_split(source)
    validation = pd.concat([splits.train.iloc[[-1]], splits.validation.iloc[1:]])
    with pytest.raises(ValueError, match="overlap between splits"):
        validate_splits(source, replace(splits, validation=validation))


@pytest.mark.parametrize("corruption", ["missing_row", "unknown_id", "empty_split"])
def test_validator_rejects_incomplete_or_foreign_rows(source, corruption) -> None:
    splits = chronological_split(source)
    test = splits.test.copy()
    if corruption == "missing_row":
        test = test.iloc[:-1]
    elif corruption == "empty_split":
        test = test.iloc[:0]
    else:
        test.iloc[-1, test.columns.get_loc("TransactionID")] = 999999
    with pytest.raises(ValueError, match="every source|empty"):
        validate_splits(source, replace(splits, test=test))


def test_validator_rejects_changed_labels_for_same_id(source) -> None:
    splits = chronological_split(source)
    splits.validation.loc[70, "isFraud"] = 1 - splits.validation.loc[70, "isFraud"]
    with pytest.raises(ValueError, match="isFraud labels changed"):
        validate_splits(source, splits)


def test_validator_rejects_changed_times_for_same_id(source) -> None:
    splits = chronological_split(source)
    splits.test.loc[99, "TransactionDT"] += 10
    with pytest.raises(ValueError, match="TransactionDT changed"):
        validate_splits(source, splits)


def test_validator_rejects_unsorted_split(source) -> None:
    splits = chronological_split(source)
    with pytest.raises(ValueError, match="not chronologically ordered"):
        validate_splits(source, replace(splits, train=splits.train.iloc[::-1]))


@pytest.mark.parametrize("kind", ["same_timestamp", "future_in_train"])
def test_validator_detects_time_leakage_even_with_disjoint_ids(source, kind) -> None:
    if kind == "same_timestamp":
        source.loc[70, "TransactionDT"] = source.loc[69, "TransactionDT"]
        # A naive row-position 70/15/15 cut incorrectly separates equal times.
        splits = ChronologicalSplits(source.iloc[:70], source.iloc[70:85], source.iloc[85:])
    else:
        splits = ChronologicalSplits(
            pd.concat([source.iloc[:55], source.iloc[70:85]]),
            source.iloc[55:70], source.iloc[85:],
        )
    with pytest.raises(ValueError, match="Temporal leakage"):
        validate_splits(source, splits)


def test_report_uses_all_rows_and_exports_only_aggregates(tmp_path: Path, source) -> None:
    raw, reports = tmp_path / "raw", tmp_path / "reports"
    raw.mkdir()
    path = raw / "train_transaction.csv"
    source.sample(frac=1, random_state=3).to_csv(path, index=False)
    original = path.read_bytes()
    summary = run_split_report(raw, reports)
    assert summary["rows"].tolist() == [70, 15, 15]
    assert summary["fraud_rate_pct"].tolist() == [20.0, 20.0, 20.0]
    assert path.read_bytes() == original
    assert len(list(reports.iterdir())) == 3
    audit = json.loads((reports / "chronological_split_summary.json").read_text())
    assert audit["source_rows"] == 100 and audit["source_fraud_rows"] == 20
    assert all(audit["integrity"].values())
    assert not audit["timestamp_group_adjustment"]
    csv = pd.read_csv(reports / "chronological_split_summary.csv")
    assert "TransactionID" not in csv.columns and "isFraud" not in csv.columns
    assert len(csv) == 3
    assert "integrity PASS" in (reports / "chronological_split_summary.md").read_text()


def test_missing_file_is_graceful_and_does_not_overwrite_report(tmp_path: Path, capsys) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    previous = reports / "chronological_split_summary.md"
    previous.write_text("previous valid run", encoding="utf-8")
    assert main(["--raw-dir", str(tmp_path / "absent"), "--output-dir", str(reports)]) == 2
    error = capsys.readouterr().err
    assert "train_transaction.csv" in error and "Traceback" not in error
    assert previous.read_text() == "previous valid run"
