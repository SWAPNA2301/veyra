"""Chronological 70/15/15 holdouts; no preprocessing, sampling or model fitting.

Run the aggregate-only audit with ``python -m veyra.splitting``. The reusable
splitter accepts either a transaction table or an already joined DataFrame.
"""

from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from .data import JOIN_KEY, TARGET_COLUMN, TIME_COLUMN, check_schema, load_csv

SPLIT_NAMES = ("train", "validation", "test")
TARGET_PERCENTAGES = (70, 15, 15)
SPLIT_COLUMNS = (JOIN_KEY, TIME_COLUMN, TARGET_COLUMN)


@dataclass(frozen=True)
class ChronologicalSplits:
    """Independent DataFrames with original columns, dtypes and row indices."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def items(self) -> tuple[tuple[str, pd.DataFrame], ...]:
        return tuple((name, getattr(self, name)) for name in SPLIT_NAMES)


def _validate_input(frame: pd.DataFrame) -> None:
    schema = check_schema(frame, SPLIT_COLUMNS)
    if not schema.passed:
        raise ValueError(
            f"Invalid split schema: missing={schema.missing_required_columns}, "
            f"duplicate_columns={schema.duplicate_columns}"
        )
    if frame.empty:
        raise ValueError("Cannot split an empty transaction table")
    if frame[JOIN_KEY].isna().any():
        raise ValueError(f"{JOIN_KEY} must not contain missing values")
    if frame[JOIN_KEY].duplicated().any():
        raise ValueError(f"{JOIN_KEY} must be unique; duplicate IDs are not deduplicated")
    time = frame[TIME_COLUMN]
    if (
        not pd.api.types.is_numeric_dtype(time)
        or pd.api.types.is_bool_dtype(time)
        or pd.api.types.is_complex_dtype(time)
    ):
        raise ValueError(f"{TIME_COLUMN} must be numeric elapsed seconds")
    if time.isna().any() or time.isin([float("inf"), float("-inf")]).any() or (time < 0).any():
        raise ValueError(f"{TIME_COLUMN} must be finite, nonnegative and non-null")
    labels = frame[TARGET_COLUMN]
    if labels.isna().any() or not labels.isin([0, 1]).all() or pd.api.types.is_complex_dtype(labels):
        raise ValueError(f"{TARGET_COLUMN} must contain only non-null binary labels 0/1")


def _nearest_boundary(boundaries: list[int], target: int) -> int:
    """Choose the nearest valid gap, preferring the earlier gap on a tie."""
    position = bisect_left(boundaries, target)
    candidates = boundaries[max(0, position - 1):position + 1]
    return min(candidates, key=lambda boundary: (abs(boundary - target), boundary))


def chronological_split(frame: pd.DataFrame) -> ChronologicalSplits:
    """Split by elapsed time into approximately 70/15/15 percent of rows.

    Use a stable ascending sort of TransactionDT only. Nominal cut positions
    are floor(70*n/100) and floor(85*n/100), with the remainder in test. Select
    the nearest complete-timestamp gap to the first cut, reserving at least
    one timestamp group for validation and one for test; then select the
    nearest remaining gap to the second cut. Equal-distance gaps go earlier.

    Every timestamp belongs to exactly one split, so train.max_time is strictly
    less than validation.min_time, and validation.max_time < test.min_time.
    Large timestamp groups can substantially alter the target proportions.
    Fewer than three distinct timestamps cannot form three nonempty holdouts.

    Labels and features never influence boundary selection. They are neither
    fitted nor transformed. Labels may have different prevalence across time;
    a single-class split is valid and is reported, not resampled.
    """
    _validate_input(frame)
    ordered = frame.sort_values(TIME_COLUMN, kind="mergesort")
    group_sizes = ordered.groupby(TIME_COLUMN, sort=False, observed=True).size()
    if len(group_sizes) < 3:
        raise ValueError("At least three distinct TransactionDT values are required for nonempty splits")
    boundaries = group_sizes.cumsum().iloc[:-1].tolist()
    train_end = _nearest_boundary(boundaries[:-1], len(ordered) * 70 // 100)
    remaining = boundaries[bisect_left(boundaries, train_end) + 1:]
    validation_end = _nearest_boundary(remaining, len(ordered) * 85 // 100)
    splits = ChronologicalSplits(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:validation_end].copy(),
        test=ordered.iloc[validation_end:].copy(),
    )
    validate_splits(frame, splits)
    return splits


def validate_splits(source: pd.DataFrame, splits: ChronologicalSplits) -> None:
    """Raise ValueError on overlap, loss, time leakage or changed labels/times.

    Compare rows by TransactionID, never by DataFrame index or row position.
    This validator can be rerun after joins/transformations that should have
    preserved the split's schema, identifiers, labels and time information.
    """
    _validate_input(source)
    for name, part in splits.items():
        if part.empty:
            raise ValueError(f"{name} split is empty")
        _validate_input(part)
        if not part.columns.equals(source.columns) or not part.dtypes.equals(source.dtypes):
            raise ValueError(f"{name} split changed source columns or dtypes")
        if not part[TIME_COLUMN].is_monotonic_increasing:
            raise ValueError(f"{name} split is not chronologically ordered")

    # Only core columns are concatenated for validation, even for wide inputs.
    combined = pd.concat([part.loc[:, list(SPLIT_COLUMNS)] for _, part in splits.items()], ignore_index=True)
    if combined[JOIN_KEY].duplicated().any():
        raise ValueError("TransactionID overlap between splits")
    if len(combined) != len(source) or not combined[JOIN_KEY].isin(source[JOIN_KEY]).all():
        raise ValueError("Splits must cover every source TransactionID exactly once")
    observed = combined.set_index(JOIN_KEY)
    expected = source.loc[:, list(SPLIT_COLUMNS)].set_index(JOIN_KEY).reindex(observed.index)
    if not observed[TARGET_COLUMN].equals(expected[TARGET_COLUMN]):
        raise ValueError("isFraud labels changed for one or more TransactionID values")
    if not observed[TIME_COLUMN].equals(expected[TIME_COLUMN]):
        raise ValueError("TransactionDT changed for one or more TransactionID values")
    for earlier, later in ((splits.train, splits.validation), (splits.validation, splits.test)):
        if not earlier[TIME_COLUMN].iloc[-1] < later[TIME_COLUMN].iloc[0]:
            raise ValueError("Temporal leakage: split time ranges overlap or share a timestamp")


def summarize_splits(splits: ChronologicalSplits) -> pd.DataFrame:
    """Return aggregate counts, label prevalence and elapsed-time boundaries."""
    total = sum(len(part) for _, part in splits.items())
    target_train = total * 70 // 100
    target_validation_end = total * 85 // 100
    nominal_counts = (target_train, target_validation_end - target_train, total - target_validation_end)
    rows = []
    for (name, part), target_pct, target_rows in zip(splits.items(), TARGET_PERCENTAGES, nominal_counts):
        if part.empty:
            raise ValueError(f"{name} split is empty")
        fraud_count = int((part[TARGET_COLUMN] == 1).sum())
        rows.append({
            "split": name,
            "target_pct": target_pct,
            "target_rows": target_rows,
            "rows": len(part),
            "actual_pct": 100 * len(part) / total,
            "fraud_rows": fraud_count,
            "non_fraud_rows": len(part) - fraud_count,
            "fraud_rate_pct": 100 * fraud_count / len(part),
            "TransactionDT_min": part[TIME_COLUMN].iloc[0],
            "TransactionDT_max": part[TIME_COLUMN].iloc[-1],
        })
    return pd.DataFrame(rows)


def run_split_report(raw_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Audit membership using all input rows, loading only the three core columns.

    Feature columns cannot affect membership. For downstream use call
    chronological_split on the complete or one-to-one-joined DataFrame.
    No transaction-level rows or ID manifests are exported by this command.
    """
    source_path = raw_dir / "train_transaction.csv"
    frame = load_csv(source_path, columns=list(SPLIT_COLUMNS))
    splits = chronological_split(frame)
    summary = summarize_splits(splits)
    generated = datetime.now(timezone.utc).isoformat()
    adjusted = bool((summary["rows"] != summary["target_rows"]).any())
    payload = {
        "generated_at_utc": generated,
        "source": source_path.name,
        "source_bytes": source_path.stat().st_size,
        "source_rows": len(frame),
        "source_fraud_rows": int((frame[TARGET_COLUMN] == 1).sum()),
        "target_percentages": list(TARGET_PERCENTAGES),
        "timestamp_group_adjustment": adjusted,
        "boundary_policy": "nearest admissible timestamp gaps to floor(0.70*n), then floor(0.85*n); ties go earlier",
        "integrity": {
            "unique_non_null_ids": True,
            "disjoint_and_exhaustive": True,
            "strict_chronological_boundaries": True,
            "no_shared_timestamps": True,
            "labels_and_times_preserved_by_id": True,
        },
        "splits": summary.to_dict(orient="records"),
    }
    lines = [
        "# Chronological train / validation / test split", "",
        f"Generated (UTC): {generated}. All {len(frame):,} transaction rows checked; **integrity PASS**.", "",
        "| Split | Target rows | Actual rows | Actual % | Fraud rows | Fraud rate % | TransactionDT min | TransactionDT max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['split']} | {row['target_rows']:,} | {row['rows']:,} | {row['actual_pct']:.6f} "
            f"| {row['fraud_rows']:,} | {row['fraud_rate_pct']:.4f} "
            f"| {row['TransactionDT_min']:,} | {row['TransactionDT_max']:,} |"
        )
    lines.extend([
        "", "## Policy and integrity", "",
        "- Targets are 70/15/15 of rows, not elapsed-time duration; no random split or label stratification.",
        "- Stable ascending TransactionDT sort. Nominal cuts are floor(0.70*n) and floor(0.85*n). "
        "Each cut uses the nearest admissible complete-timestamp gap; equal-distance gaps go earlier. "
        "At least one timestamp group is reserved for each remaining split.",
        f"- Timestamp-group adjustment required: {'yes' if adjusted else 'no'}. "
        "Large tied groups can change the target proportions; fewer than three distinct timestamps is rejected.",
        "- IDs are non-null, unique within and disjoint across splits; every source row appears exactly once.",
        "- Every split is ordered. max(train.TransactionDT) < min(validation.TransactionDT), "
        "and max(validation.TransactionDT) < min(test.TransactionDT); equal timestamps never cross boundaries.",
        "- Fraud labels are binary, non-null and unchanged per TransactionID; TransactionDT is also unchanged. "
        "Fraud prevalence is measured after membership is fixed and is not forced to match across periods.",
        "- This audit reads only the three split-defining columns. The same splitter accepts all transaction/identity "
        "features, preserving original columns, dtypes and indices. No raw rows or ID manifests were saved.", "",
        "## Downstream use", "",
        "No preprocessing or model was fitted. Future imputers, encoders, scalers and feature selection must be "
        "fitted only on training data, then applied unchanged to validation and test. Time-derived or aggregate "
        "features must not use future transactions; keep the target out of features. Strict timestamp boundaries "
        "do not by themselves prove every feature was available at prediction time.", "",
        "Validation is for development; preserve test as the final holdout. Test-label prevalence here is only an "
        "integrity report, not a basis for changing the split or selecting features. Earlier full-data EDA reports "
        "are descriptive, not training-fitted transformations.", "",
        "[Aggregate CSV](chronological_split_summary.csv) | [Machine-readable audit](chronological_split_summary.json)",
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "chronological_split_summary.csv", index=False)
    (output_dir / "chronological_split_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "chronological_split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    try:
        summary = run_split_report(args.raw_dir, args.output_dir)
    except (OSError, ValueError, duckdb.Error) as exc:
        print(f"Split audit failed: {exc}. No new successful split report was produced.", file=sys.stderr)
        return 2
    print(summary[["split", "rows", "actual_pct", "fraud_rows", "fraud_rate_pct"]].to_string(index=False))
    print(f"Integrity PASS; aggregate reports saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
