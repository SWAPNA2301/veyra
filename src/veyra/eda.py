"""Reproducible, aggregate-only EDA: ``python -m veyra.eda``."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .data import (
    JOIN_KEY,
    TARGET_COLUMN,
    TIME_COLUMN,
    CsvBackend,
    DataPaths,
    check_schema,
    find_duplicate_rows,
    fraud_class_distribution,
    load_veyra_data,
    merge_transaction_identity,
    missing_value_profile,
    profile_duplicates,
    validate_input_schemas,
)

ANALYSES = {
    "01_product_risk": "Product risk and monetary exposure",
    "02_amount_bands": "Transaction amount bands",
    "03_time_trends": "Relative-week fraud and coverage trends",
    "04_identity_device": "Identity coverage and device type",
    "05_email_domains": "Purchaser-email domains (at least 100 transactions)",
}


@dataclass
class EdaResult:
    quality: dict[str, Any]
    missing_values: pd.DataFrame
    analyses: dict[str, pd.DataFrame]


def _table_quality(frame: pd.DataFrame) -> dict[str, int]:
    keys = profile_duplicates(frame, subset=[JOIN_KEY])
    # Exact duplicates must share their key. Restrict the expensive wide-row
    # comparison to duplicate-key candidates, without skipping any candidates.
    candidates = find_duplicate_rows(frame, subset=[JOIN_KEY])
    exact = profile_duplicates(candidates)
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "distinct_ids": int(frame[JOIN_KEY].nunique()),
        "null_ids": int(frame[JOIN_KEY].isna().sum()),
        "duplicate_key_rows": keys.duplicate_row_count,
        "duplicate_key_groups": keys.duplicate_group_count,
        "exact_duplicate_rows": exact.duplicate_row_count,
        "exact_duplicate_groups": exact.duplicate_group_count,
    }


def _numeric_quality(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    infinite = numeric.isin([float("inf"), float("-inf")])
    finite = numeric[numeric.notna() & ~infinite].astype(float)
    return {
        "missing": int(series.isna().sum()),
        "non_numeric": int((series.notna() & numeric.isna()).sum()),
        "non_finite": int(infinite.sum()),
        "negative": int((finite < 0).sum()),
        "zero": int((finite == 0).sum()),
        "finite_count": len(finite),
        "min": float(finite.min()) if len(finite) else None,
        "median": float(finite.median()) if len(finite) else None,
        "mean": float(finite.mean()) if len(finite) else None,
        "p95": float(finite.quantile(0.95)) if len(finite) else None,
        "p99": float(finite.quantile(0.99)) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
        "sum": float(finite.sum()),
    }


def _source_counts(paths: DataPaths) -> dict[str, int]:
    """Independently count CSV records with DuckDB, not physical text lines."""
    counts = {}
    with duckdb.connect() as connection:
        for name, path in (("transactions", paths.transaction), ("identity", paths.identity)):
            with path.open(encoding="utf-8-sig", newline="") as source:
                header = next(csv.reader(source), [])
            if not header or len(header) != len(set(header)):
                raise ValueError(f"{path.name}: empty or duplicate CSV column names")
            counts[name] = connection.execute(
                "SELECT COUNT(*) FROM read_csv(?, header=true, delim=',', "
                "all_varchar=true, ignore_errors=false)",
                [str(path)],
            ).fetchone()[0]
    return counts


def analyze_frames(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
    *,
    source_row_counts: dict[str, int] | None = None,
) -> EdaResult:
    """Validate full input tables, reuse the existing merge/profilers, run SQL.

    Invalid keys/labels/amounts/times fail closed before running analyses.
    Duplicate counts include *all* rows in each duplicate group. No rows are
    deduplicated, imputed, filtered, or sorted in the source or merged tables.
    """
    validate_input_schemas(transactions, identity)
    for name, frame, required in (
        ("transactions", transactions, {"TransactionAmt", "ProductCD", "P_emaildomain"}),
        ("identity", identity, {"DeviceType"}),
    ):
        schema = check_schema(frame, required, dataset=name)
        if not schema.passed:
            raise ValueError(f"{name}: missing EDA columns {schema.missing_required_columns}")
    overlap = set(transactions.columns).intersection(identity.columns) - {JOIN_KEY}
    if overlap:
        raise ValueError(f"Ambiguous overlapping input columns: {sorted(overlap)}")

    tables = {
        "transactions": _table_quality(transactions),
        "identity": _table_quality(identity),
    }
    amount = _numeric_quality(transactions["TransactionAmt"])
    time = _numeric_quality(transactions[TIME_COLUMN])
    labels = transactions[TARGET_COLUMN]
    errors = []
    if transactions.empty:
        errors.append("transaction table is empty")
    for name, stats in tables.items():
        if stats["null_ids"] or stats["duplicate_key_rows"]:
            errors.append(
                f"{name}: {stats['null_ids']} null IDs, "
                f"{stats['duplicate_key_rows']} duplicate-key rows, "
                f"{stats['exact_duplicate_rows']} exact-duplicate rows"
            )
        if source_row_counts is not None:
            stats["csv_rows"] = source_row_counts[name]
            if stats["rows"] != stats["csv_rows"]:
                errors.append(f"{name}: CSV and loaded row counts differ")
    invalid_labels = int((labels.notna() & ~labels.isin([0, 1])).sum())
    missing_labels = int(labels.isna().sum())
    if invalid_labels or missing_labels:
        errors.append(f"isFraud: {invalid_labels} invalid, {missing_labels} missing labels")
    for name, stats in (("TransactionAmt", amount), (TIME_COLUMN, time)):
        if not pd.api.types.is_numeric_dtype(transactions[name]) or pd.api.types.is_bool_dtype(transactions[name]):
            errors.append(f"{name}: expected a numeric column, got {transactions[name].dtype}")
        failures = {key: stats[key] for key in ("missing", "non_numeric", "non_finite", "negative") if stats[key]}
        if failures:
            errors.append(f"{name}: {failures}")
    if errors:
        raise ValueError("Data quality validation failed: " + "; ".join(errors))

    merged = merge_transaction_identity(transactions, identity)
    ids_preserved = merged[JOIN_KEY].reset_index(drop=True).equals(transactions[JOIN_KEY].reset_index(drop=True))
    dt_preserved = merged[TIME_COLUMN].reset_index(drop=True).equals(transactions[TIME_COLUMN].reset_index(drop=True))
    if len(merged) != len(transactions) or not ids_preserved or not dt_preserved:
        raise ValueError("Left join changed transaction count, order, or TransactionDT")
    matched = merged[JOIN_KEY].isin(identity[JOIN_KEY])
    orphan_ids = int((~identity[JOIN_KEY].isin(transactions[JOIN_KEY])).sum())
    missing_profiles = []
    missing_summary = {}
    for name, frame in (("transactions", transactions), ("identity", identity), ("merged", merged)):
        profile = missing_value_profile(frame)
        profile.insert(0, "dataset", name)
        profile.insert(2, "rows", len(frame))
        missing_profiles.append(profile)
        missing_summary[name] = {
            "columns": len(frame.columns),
            "columns_with_missing": int((profile["missing_count"] > 0).sum()),
            "columns_over_50_pct_missing": int((profile["missing_pct"] > 50).sum()),
            "all_missing_columns": int((profile["missing_pct"] == 100).sum()),
            "missing_cells": int(profile["missing_count"].sum()),
        }

    fraud_rows = int((labels == 1).sum())
    amount["fraud_amount"] = float(transactions.loc[labels == 1, "TransactionAmt"].sum())
    amount["fraud_amount_share_pct"] = 100 * amount["fraud_amount"] / amount["sum"] if amount["sum"] else None
    time["span_days"] = (time["max"] - time["min"]) / 86400
    time["backward_steps"] = int((transactions[TIME_COLUMN].diff() < 0).sum())
    warnings = []
    if orphan_ids:
        warnings.append(f"{orphan_ids} identity IDs have no transaction; the left join excludes them.")
    if amount["zero"]:
        warnings.append(f"{amount['zero']} zero-amount transactions retained for review.")
    if time["backward_steps"]:
        warnings.append("Source transactions are not chronological; sort by TransactionDT when defining future splits.")
    if identity.empty:
        warnings.append("Identity table is empty; all transactions are retained with missing identity fields.")

    quality = {
        "status": "passed",
        "tables": tables,
        "join": {
            "type": "left", "validated_cardinality": "one_to_one",
            "rows": len(merged), "columns": len(merged.columns),
            "matched_transactions": int(matched.sum()),
            "unmatched_transactions": int((~matched).sum()),
            "identity_coverage_pct": float(100 * matched.mean()),
            "orphan_identity_ids": orphan_ids,
            "transaction_order_preserved": ids_preserved,
            "transaction_dt_preserved": dt_preserved,
        },
        "fraud": {
            "fraud_transactions": fraud_rows, "non_fraud_transactions": len(labels) - fraud_rows,
            "fraud_rate_pct": 100 * fraud_rows / len(labels),
            "missing_labels": missing_labels, "invalid_labels": invalid_labels,
            "distribution": fraud_class_distribution(transactions).to_dict(orient="records"),
        },
        "amount": amount, "time": time, "missing": missing_summary, "warnings": warnings,
    }
    # Only the seven analysis columns are scanned by SQL, not all ~430 features.
    analysis_data = merged[[TARGET_COLUMN, TIME_COLUMN, "TransactionAmt", "ProductCD", "P_emaildomain", "DeviceType"]].copy()
    analysis_data[TARGET_COLUMN] = analysis_data[TARGET_COLUMN].astype("int64")
    analysis_data["has_identity"] = matched.to_numpy()
    analyses = {}
    with duckdb.connect() as connection:
        connection.register("analysis_data", analysis_data)
        for name in ANALYSES:
            query = files("veyra").joinpath("sql", f"{name}.sql").read_text(encoding="utf-8")
            analyses[name] = connection.execute(query).df()
    return EdaResult(quality, pd.concat(missing_profiles, ignore_index=True), analyses)


def _markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:,.4f}".rstrip("0").rstrip(".")
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_reports(result: EdaResult, output_dir: Path) -> None:
    """Persist aggregate results only; never export merged or individual rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    q = result.quality
    table_rows = pd.DataFrame.from_dict(q["tables"], orient="index").rename_axis("dataset").reset_index()
    missing_rows = pd.DataFrame.from_dict(q["missing"], orient="index").rename_axis("dataset").reset_index()
    join, fraud, amount, time = (q[name] for name in ("join", "fraud", "amount", "time"))
    quality_lines = [
        "# IEEE-CIS data quality summary", "",
        f"Generated (UTC): {q['generated_at_utc']}. Full files, no sampling. Status: **PASS**.", "",
        "## Rows, schema and duplicates", "", _markdown_table(table_rows), "",
        "CSV row counts are independently parsed by DuckDB and equal the loaded counts. "
        "Required columns and unique CSV headers pass. Duplicate-row counts include every row "
        "in a repeated group; exact duplicates include TransactionID. Unique IDs imply zero "
        "exact duplicates; otherwise full-row checks examine every duplicate-key candidate.", "",
        "## Join and labels", "",
        f"- Validated one-to-one left join: {join['rows']:,} rows, {join['columns']} columns; no row multiplication or loss.",
        f"- Matched: {join['matched_transactions']:,} ({join['identity_coverage_pct']:.4f}%); "
        f"unmatched: {join['unmatched_transactions']:,}; orphan identity IDs: {join['orphan_identity_ids']:,}.",
        "- Transaction order and original TransactionDT values/dtype are preserved.",
        f"- Fraud: {fraud['fraud_transactions']:,} / {join['rows']:,} = **{fraud['fraud_rate_pct']:.4f}%**; "
        f"non-fraud: {fraud['non_fraud_transactions']:,}; missing/invalid labels: 0/0.", "",
        "## Transaction amounts and time", "",
        "Amounts are in recorded dataset units; no currency assumption, rounding of input, or outlier removal.", "",
        _markdown_table(pd.DataFrame([{key: amount[key] for key in ("min", "median", "mean", "p95", "p99", "max", "sum")} ])), "",
        f"- Amount checks: missing={amount['missing']}, non-numeric={amount['non_numeric']}, "
        f"non-finite={amount['non_finite']}, negative={amount['negative']}, zero={amount['zero']}.",
        f"- Fraud-labelled amount: {amount['fraud_amount']:,.2f}; "
        f"share of total amount: {_markdown_amount_share(amount['fraud_amount_share_pct'])}. "
        "This is labelled transaction value, not measured financial loss.",
        f"- TransactionDT range: {time['min']:,.0f}–{time['max']:,.0f} seconds; "
        f"observed span: {time['span_days']:.4f} days; backward steps: {time['backward_steps']}.",
        "- No missing, non-numeric, non-finite or negative TransactionDT values. "
        "Future splits must use time order; no split or model is created here.", "",
        "## Missing values", "", _markdown_table(missing_rows), "",
        "Most-missing merged columns (full per-column profile in [missing_values.csv](missing_values.csv)):", "",
        _markdown_table(result.missing_values.loc[result.missing_values["dataset"] == "merged"].head(10)), "",
        "Identity-table missingness describes available records; merged missingness also includes "
        "transactions without any identity match. No imputation or row dropping was performed.", "",
        "## Warnings", "", *(q["warnings"] or ["None from the structural/value checks. Class imbalance and high missingness remain analytical considerations."]), "",
        "See [SQL analyses](sql_analyses.md) and [machine-readable quality results](data_quality.json).", "",
        f"Runtime: Python {q['versions']['python']}, Pandas {q['versions']['pandas']}, DuckDB {q['versions']['duckdb']}.",
    ]
    sql_lines = [
        "# IEEE-CIS: five DuckDB analyses", "",
        f"Generated (UTC): {q['generated_at_utc']}. Full validated left join: {join['rows']:,} transactions; "
        f"baseline fraud rate **{fraud['fraud_rate_pct']:.4f}%**.", "",
        "Rates are percentages of transactions within each displayed group. These are in-sample, "
        "descriptive associations, not evidence of causation or validated model performance. "
        "Amounts are dataset units. All SQL and complete aggregate results accompany this report.",
    ]
    for name, title in ANALYSES.items():
        frame = result.analyses[name]
        query = files("veyra").joinpath("sql", f"{name}.sql").read_text(encoding="utf-8")
        (output_dir / f"{name}.sql").write_text(query, encoding="utf-8")
        frame.to_csv(output_dir / f"{name}.csv", index=False)
        sql_lines.extend(["", f"## {title}", "", f"[SQL]({name}.sql) · [All {len(frame)} result groups]({name}.csv)", ""])
        if frame.empty:
            sql_lines.append("No groups meet this analysis's criteria.")
            continue
        highest = frame.loc[frame["fraud_rate_pct"].idxmax()]
        lowest = frame.loc[frame["fraud_rate_pct"].idxmin()]
        group_column = frame.columns[0]
        sql_lines.append(
            f"Highest observed rate: **{highest[group_column]}**, {highest['fraud_rate_pct']:.4f}% "
            f"({int(highest['fraud_transactions']):,}/{int(highest['transactions']):,}); "
            f"lowest: {lowest[group_column]}, {lowest['fraud_rate_pct']:.4f}% "
            f"({int(lowest['fraud_transactions']):,}/{int(lowest['transactions']):,})."
        )
        preview = frame.head(10)
        if name == "03_time_trends":
            sql_lines.append("Weeks are floor(TransactionDT / 604800), not calendar weeks. "
                             "First/last bins may be partial. Preview shows the first/last three observed bins.")
            preview = pd.concat([frame.head(3), frame.tail(3)]).drop_duplicates()
        if name == "04_identity_device":
            sql_lines.append("A matched record with missing DeviceType is distinct from no identity record.")
        if name == "05_email_domains":
            retained = int(frame["transactions"].sum())
            sql_lines.append(f"Minimum support: 100 transactions per domain (including a missing group). "
                             f"Retained {retained:,} of {join['rows']:,} transactions; excluded {join['rows'] - retained:,} "
                             "in smaller groups. Preview is the ten highest observed rates, not a risk ranking for deployment.")
        sql_lines.extend(["", _markdown_table(preview)])
    (output_dir / "data_quality.md").write_text("\n".join(quality_lines) + "\n", encoding="utf-8")
    (output_dir / "sql_analyses.md").write_text("\n".join(sql_lines) + "\n", encoding="utf-8")
    (output_dir / "data_quality.json").write_text(json.dumps(q, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result.missing_values.to_csv(output_dir / "missing_values.csv", index=False)


def _markdown_amount_share(value: float | None) -> str:
    return "undefined (zero total amount)" if value is None else f"{value:.4f}%"


def run_eda(raw_dir: Path, output_dir: Path, *, backend: CsvBackend = "pandas") -> EdaResult:
    paths = DataPaths.from_raw_dir(raw_dir)
    absent = [path.name for path in (paths.transaction, paths.identity) if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Missing input file(s) in {raw_dir}: {', '.join(absent)}. Place both IEEE-CIS training CSVs there and retry.")
    source_counts = _source_counts(paths)
    transactions, identity = load_veyra_data(paths, backend=backend)
    result = analyze_frames(transactions, identity, source_row_counts=source_counts)
    result.quality.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "sources": {path.name: {"bytes": path.stat().st_size} for path in (paths.transaction, paths.identity)},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "duckdb": duckdb.__version__},
    })
    write_reports(result, output_dir)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--backend", choices=["pandas", "duckdb"], default="pandas")
    args = parser.parse_args(argv)
    try:
        result = run_eda(args.raw_dir, args.output_dir, backend=args.backend)
    except (OSError, ValueError, duckdb.Error, pd.errors.ParserError) as exc:
        print(f"EDA failed: {exc}\nNo new successful EDA summary was produced; previous reports, if any, are not results of this run.", file=sys.stderr)
        return 2
    q = result.quality
    print(f"PASS: {q['tables']['transactions']['rows']:,} transactions, {q['tables']['identity']['rows']:,} identity rows; "
          f"fraud rate {q['fraud']['fraud_rate_pct']:.4f}%; five SQL analyses saved to {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
