"""Load only the existing train/validation windows, never a test DataFrame."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .data import DataPaths, JOIN_KEY, TARGET_COLUMN, TIME_COLUMN, merge_transaction_identity
from .features import ALLOWED_FEATURES, NUMERIC_FEATURES, exclusion_reason


@dataclass
class DevelopmentData:
    train: pd.DataFrame
    validation: pd.DataFrame
    provenance: dict


def validate_development_data(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    """Validate temporal integrity and labels without accepting a test argument."""
    for name, frame in (("train", train), ("validation", validation)):
        required = {JOIN_KEY, TIME_COLUMN, TARGET_COLUMN, "TransactionAmt"}
        if frame.columns.duplicated().any() or not required.issubset(frame.columns):
            raise ValueError(f"{name}: duplicate or missing required columns")
        if frame.empty or frame[JOIN_KEY].isna().any() or frame[JOIN_KEY].duplicated().any():
            raise ValueError(f"{name}: empty table or missing/duplicate TransactionID")
        time = frame[TIME_COLUMN]
        if not pd.api.types.is_numeric_dtype(time) or time.isna().any() or not np.isfinite(time).all() or (time < 0).any():
            raise ValueError(f"{name}: invalid TransactionDT")
        if not time.is_monotonic_increasing:
            raise ValueError(f"{name}: TransactionDT is not ordered")
        labels = frame[TARGET_COLUMN]
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(f"{name}: invalid isFraud labels")
        if labels.nunique() != 2:
            raise ValueError(f"{name}: both fraud classes are required for fitting/ranking metrics")
    if train[JOIN_KEY].isin(validation[JOIN_KEY]).any():
        raise ValueError("TransactionID overlap between train and validation")
    if train[TIME_COLUMN].max() >= validation[TIME_COLUMN].min():
        raise ValueError("Temporal leakage between train and validation")
    if not train.columns.equals(validation.columns):
        raise ValueError("Train and validation schemas differ")


def _header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        columns = next(csv.reader(source), [])
    if not columns or len(columns) != len(set(columns)):
        raise ValueError(f"{path.name}: empty or duplicate CSV header")
    return columns


def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def load_development_data(raw_dir: Path, split_report: Path) -> DevelopmentData:
    """Reuse frozen split bounds; project/filter CSVs before materializing rows.

    CSV tokenization still scans the shared input files. Test features/labels
    are not returned, validated, profiled, encoded, fitted, predicted or scored.
    all_varchar avoids fitting a type inference rule to the holdout values.
    """
    paths = DataPaths.from_raw_dir(raw_dir)
    for path in (paths.transaction, paths.identity, split_report):
        if not path.is_file():
            raise FileNotFoundError(f"Required training input not found: {path}")
    report_bytes = split_report.read_bytes()
    report = json.loads(report_bytes)
    if report.get("target_percentages") != [70, 15, 15] or not report.get("integrity") or not all(report["integrity"].values()):
        raise ValueError("A successful existing 70/15/15 split audit is required")
    if report.get("source_bytes") != paths.transaction.stat().st_size:
        raise ValueError("Transaction CSV size changed since the split audit; regenerate/review the split first")
    # Intentionally retain no test-label statistics from the prior audit.
    windows = {row["split"]: row for row in report["splits"] if row["split"] in {"train", "validation"}}
    if set(windows) != {"train", "validation"}:
        raise ValueError("Split audit is missing train/validation windows")
    left, right = windows["train"], windows["validation"]
    if not left["TransactionDT_min"] <= left["TransactionDT_max"] < right["TransactionDT_min"] <= right["TransactionDT_max"]:
        raise ValueError("Invalid frozen chronological boundaries")

    transaction_columns, identity_columns = _header(paths.transaction), _header(paths.identity)
    if not {JOIN_KEY, TARGET_COLUMN, TIME_COLUMN, "TransactionAmt"}.issubset(transaction_columns) or JOIN_KEY not in identity_columns:
        raise ValueError("Raw CSVs are missing required training columns")
    transaction_features = [name for name in ALLOWED_FEATURES if name in transaction_columns]
    identity_features = [name for name in ALLOWED_FEATURES if name in identity_columns]
    if set(transaction_features) & set(identity_features):
        raise ValueError("Ambiguous overlapping transaction and identity features")
    projections = [_quoted(JOIN_KEY)]
    for name in [TIME_COLUMN, TARGET_COLUMN, *transaction_features]:
        quoted = _quoted(name)
        projections.append(f"CAST({quoted} AS DOUBLE) AS {quoted}" if name in NUMERIC_FEATURES or name in {TIME_COLUMN, TARGET_COLUMN} else quoted)
    read_options = "header=true, delim=',', all_varchar=true, ignore_errors=false"
    with duckdb.connect(config={"threads": 4, "memory_limit": "1GB"}) as connection:
        connection.execute(
            f"CREATE TEMP TABLE development AS SELECT {', '.join(projections)} "
            f"FROM read_csv(?, {read_options}) WHERE CAST(TransactionDT AS DOUBLE) BETWEEN ? AND ?",
            [str(paths.transaction), left["TransactionDT_min"], right["TransactionDT_max"]],
        )
        transactions = connection.execute("SELECT * FROM development ORDER BY TransactionDT").df()
        identity = connection.execute(
            f"SELECT {', '.join(_quoted(name) for name in [JOIN_KEY, *identity_features])} "
            f"FROM read_csv(?, {read_options}) "
            "WHERE TransactionID IN (SELECT TransactionID FROM development)",
            [str(paths.identity)],
        ).df()
    if transactions[JOIN_KEY].isna().any() or transactions[JOIN_KEY].duplicated().any():
        raise ValueError("Training-window transaction IDs must be non-null and unique")
    merged = merge_transaction_identity(transactions, identity)
    train = merged.loc[merged[TIME_COLUMN] <= left["TransactionDT_max"]].reset_index(drop=True)
    validation = merged.loc[merged[TIME_COLUMN] >= right["TransactionDT_min"]].reset_index(drop=True)
    if len(train) + len(validation) != len(merged):
        raise ValueError("New rows fall in the gap between frozen split windows")
    validate_development_data(train, validation)
    for name, frame in (("train", train), ("validation", validation)):
        expected = windows[name]
        if (
            len(frame) != expected["rows"]
            or int((frame[TARGET_COLUMN] == 1).sum()) != expected["fraud_rows"]
            or frame[TIME_COLUMN].min() != expected["TransactionDT_min"]
            or frame[TIME_COLUMN].max() != expected["TransactionDT_max"]
        ):
            raise ValueError(f"{name}: data no longer agrees with the frozen split audit")
    excluded = {name: exclusion_reason(name) for name in dict.fromkeys(transaction_columns + identity_columns) if name not in ALLOWED_FEATURES}
    return DevelopmentData(train, validation, {
        "split_audit_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "windows": windows,
        "test_evaluated": False,
        "test_materialized": False,
        "csv_policy": "Shared CSV scans; static typed projections and frozen time filters before materialization; no test features or labels consumed by ML",
        "excluded_source_columns": excluded,
    })
