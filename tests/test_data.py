from pathlib import Path

import pandas as pd
import pytest

from veyra.data import (
    DataPaths,
    check_schema,
    find_duplicate_rows,
    fraud_class_distribution,
    load_csv,
    load_veyra_data,
    merge_transaction_identity,
    missing_value_profile,
    profile_duplicates,
)


@pytest.fixture
def csv_paths(tmp_path: Path) -> DataPaths:
    transaction_path = tmp_path / "train_transaction.csv"
    identity_path = tmp_path / "train_identity.csv"
    pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "TransactionDT": [100, 200, 300],
            "TransactionAmt": [10.0, 25.0, 12.5],
        }
    ).to_csv(transaction_path, index=False)
    pd.DataFrame(
        {
            "TransactionID": [1, 3],
            "DeviceType": ["mobile", "desktop"],
        }
    ).to_csv(identity_path, index=False)
    return DataPaths(transaction=transaction_path, identity=identity_path)


def test_load_and_merge_preserves_transaction_dt(csv_paths: DataPaths) -> None:
    transactions, identity = load_veyra_data(csv_paths)
    merged = merge_transaction_identity(transactions, identity)

    assert list(merged["TransactionDT"]) == [100, 200, 300]
    assert merged["DeviceType"].iloc[0] == "mobile"
    assert pd.isna(merged["DeviceType"].iloc[1])
    assert merged["DeviceType"].iloc[2] == "desktop"
    assert len(merged) == len(transactions)


def test_duckdb_backend_loads_csv(csv_paths: DataPaths) -> None:
    frame = load_csv(csv_paths.transaction, backend="duckdb")

    assert frame.shape == (3, 4)
    assert set(frame.columns) == {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt"}


def test_schema_report_identifies_missing_columns() -> None:
    report = check_schema(pd.DataFrame({"TransactionID": [1]}), {"TransactionID", "TransactionDT"})

    assert report.passed is False
    assert report.missing_required_columns == ("TransactionDT",)


def test_quality_profiles() -> None:
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 1, 2],
            "isFraud": [0, 0, 1],
            "optional": [None, "present", None],
        }
    )

    duplicates = profile_duplicates(frame, subset=["TransactionID"])
    duplicate_rows = find_duplicate_rows(frame, subset=["TransactionID"])
    missing = missing_value_profile(frame)
    distribution = fraud_class_distribution(frame)

    assert duplicates.duplicate_row_count == 2
    assert duplicates.duplicate_group_count == 1
    assert len(duplicate_rows) == 2
    assert missing.loc[missing["column"] == "optional", "missing_count"].item() == 2
    assert distribution.set_index("class").loc[1, "count"] == 1


def test_duplicate_identity_keys_are_rejected(csv_paths: DataPaths) -> None:
    transactions, identity = load_veyra_data(csv_paths)
    duplicate_identity = pd.concat([identity, identity.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate TransactionID"):
        merge_transaction_identity(transactions, duplicate_identity)
