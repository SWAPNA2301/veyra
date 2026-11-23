"""Loading and first-pass quality checks for Veyra's tabular data.

The functions in this module deliberately stop before feature engineering or
model training.  ``TransactionDT`` is retained by every loading and merge
operation so downstream code can make chronological train/validation splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import duckdb
import pandas as pd

JOIN_KEY = "TransactionID"
TARGET_COLUMN = "isFraud"
TIME_COLUMN = "TransactionDT"

TRANSACTION_REQUIRED_COLUMNS = frozenset({JOIN_KEY, TARGET_COLUMN, TIME_COLUMN})
IDENTITY_REQUIRED_COLUMNS = frozenset({JOIN_KEY})

CsvBackend = Literal["pandas", "duckdb"]


@dataclass(frozen=True)
class DataPaths:
    """Paths to the two Kaggle-style input files used by Veyra."""

    transaction: Path
    identity: Path

    @classmethod
    def from_raw_dir(cls, raw_dir: str | Path) -> "DataPaths":
        raw_dir = Path(raw_dir)
        return cls(
            transaction=raw_dir / "train_transaction.csv",
            identity=raw_dir / "train_identity.csv",
        )


@dataclass(frozen=True)
class SchemaReport:
    """Result of checking required and duplicate column names."""

    dataset: str
    row_count: int
    column_count: int
    missing_required_columns: tuple[str, ...]
    duplicate_columns: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.missing_required_columns and not self.duplicate_columns


@dataclass(frozen=True)
class DuplicateProfile:
    """Compact duplicate-row summary for a DataFrame."""

    row_count: int
    duplicate_row_count: int
    duplicate_group_count: int


def _ensure_file(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"CSV file not found: {resolved}")
    return resolved


def load_csv(
    path: str | Path,
    *,
    backend: CsvBackend = "pandas",
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load a CSV with either Pandas or DuckDB.

    Pandas is the default because it provides predictable DataFrame semantics
    for the quality checks. DuckDB is available for larger or query-oriented
    ingestion and is exercised by the test suite.
    """

    csv_path = _ensure_file(path)
    if backend == "pandas":
        return pd.read_csv(csv_path, usecols=columns, low_memory=False)
    if backend == "duckdb":
        if columns:
            selected = ", ".join(f'"{column.replace(chr(34), chr(34) * 2)}"' for column in columns)
            query = (
                f"SELECT {selected} FROM read_csv_auto(?, HEADER = TRUE, "
                "SAMPLE_SIZE = -1, IGNORE_ERRORS = FALSE)"
            )
        else:
            query = "SELECT * FROM read_csv_auto(?, HEADER = TRUE, SAMPLE_SIZE = -1, IGNORE_ERRORS = FALSE)"
        with duckdb.connect() as connection:
            return connection.execute(query, [str(csv_path)]).df()
    raise ValueError(f"Unsupported CSV backend: {backend}")


def load_veyra_data(
    paths: DataPaths,
    *,
    backend: CsvBackend = "pandas",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load transaction and identity tables, then validate their schemas."""

    # Fail before reading the large transaction file if either input is absent.
    _ensure_file(paths.transaction)
    _ensure_file(paths.identity)
    transactions = load_csv(paths.transaction, backend=backend)
    identity = load_csv(paths.identity, backend=backend)
    validate_input_schemas(transactions, identity)
    return transactions, identity


def check_schema(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    dataset: str = "dataset",
) -> SchemaReport:
    """Return a schema report without mutating the input DataFrame."""

    required = set(required_columns)
    duplicate_columns = tuple(
        frame.columns[frame.columns.duplicated()].tolist()
    )
    return SchemaReport(
        dataset=dataset,
        row_count=len(frame),
        column_count=len(frame.columns),
        missing_required_columns=tuple(sorted(required.difference(frame.columns))),
        duplicate_columns=duplicate_columns,
    )


def validate_input_schemas(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
) -> Mapping[str, SchemaReport]:
    """Validate the minimum schemas needed for a safe merge.

    Raises:
        ValueError: if a required column is missing or a table has duplicate
            column names.
    """

    reports = {
        "transactions": check_schema(
            transactions,
            TRANSACTION_REQUIRED_COLUMNS,
            dataset="transactions",
        ),
        "identity": check_schema(
            identity,
            IDENTITY_REQUIRED_COLUMNS,
            dataset="identity",
        ),
    }
    failures = [
        report
        for report in reports.values()
        if not report.passed
    ]
    if failures:
        details = "; ".join(
            f"{report.dataset}: missing={list(report.missing_required_columns)}, "
            f"duplicate_columns={list(report.duplicate_columns)}"
            for report in failures
        )
        raise ValueError(f"Input schema validation failed: {details}")
    return reports


def merge_transaction_identity(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
    *,
    how: Literal["left", "inner"] = "left",
) -> pd.DataFrame:
    """Merge identity attributes onto transactions by ``TransactionID``.

    The merge is many-to-one: duplicate identity keys would multiply rows and
    are rejected. The transaction table's ``TransactionDT`` column is left
    untouched for future chronological splitting.
    """

    validate_input_schemas(transactions, identity)
    if identity[JOIN_KEY].duplicated().any():
        raise ValueError(f"Identity table contains duplicate {JOIN_KEY} values")
    return transactions.merge(
        identity,
        on=JOIN_KEY,
        how=how,
        suffixes=("", "_identity"),
        validate="many_to_one",
    )


def missing_value_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages for every column."""

    row_count = len(frame)
    profile = pd.DataFrame(
        {
            "column": frame.columns,
            "missing_count": frame.isna().sum().astype("int64").to_numpy(),
        }
    )
    profile["missing_pct"] = (
        profile["missing_count"].div(row_count).mul(100).round(4)
        if row_count
        else 0.0
    )
    return profile.sort_values(
        ["missing_count", "column"],
        ascending=[False, True],
        ignore_index=True,
    )


def profile_duplicates(
    frame: pd.DataFrame,
    *,
    subset: Sequence[str] | None = None,
) -> DuplicateProfile:
    """Summarize duplicate rows, optionally using a key subset."""

    duplicate_mask = frame.duplicated(subset=list(subset) if subset else None, keep=False)
    duplicate_rows = frame.loc[duplicate_mask]
    if duplicate_rows.empty:
        duplicate_group_count = 0
    else:
        duplicate_group_count = int(
            duplicate_rows.groupby(list(subset) if subset else list(frame.columns), dropna=False)
            .ngroups
        )
    return DuplicateProfile(
        row_count=len(frame),
        duplicate_row_count=int(duplicate_mask.sum()),
        duplicate_group_count=duplicate_group_count,
    )


def find_duplicate_rows(
    frame: pd.DataFrame,
    *,
    subset: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return all rows belonging to a duplicate group."""

    return frame.loc[
        frame.duplicated(subset=list(subset) if subset else None, keep=False)
    ].copy()


def fraud_class_distribution(
    transactions: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Return fraud class counts and percentages, including missing labels."""

    if target_column not in transactions.columns:
        raise ValueError(f"Missing fraud target column: {target_column}")
    counts = transactions[target_column].value_counts(dropna=False).rename_axis("class").reset_index(name="count")
    total = int(counts["count"].sum())
    counts["pct"] = counts["count"].div(total).mul(100).round(4) if total else 0.0
    return counts
