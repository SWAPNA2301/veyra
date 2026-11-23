from pathlib import Path

import pandas as pd
import pytest

from veyra.data import DataPaths, load_veyra_data
from veyra.eda import ANALYSES, _table_quality, analyze_frames, main, run_eda


@pytest.fixture
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = pd.DataFrame({
        "TransactionID": [6, 1, 3, 2, 5, 4],
        "isFraud": [0, 1, 0, 0, 0, 1],
        "TransactionDT": [1209601, 1, 604800, 2, 604801, 1209602],
        "TransactionAmt": [10.0, 50.0, 100.0, 250.0, 500.0, 1000.0],
        "ProductCD": ["W", "C", "W", "R", "R", "C"],
        "P_emaildomain": ["example.test", None, "example.test", None, "example.test", "example.test"],
    })
    identity = pd.DataFrame({
        "TransactionID": [6, 3, 5, 99],
        "DeviceType": ["mobile", None, "desktop", "mobile"],
    })
    return transactions, identity


def test_quality_and_all_five_queries(frames) -> None:
    transactions, identity = frames
    original = transactions.copy(deep=True)
    result = analyze_frames(transactions, identity, source_row_counts={"transactions": 6, "identity": 4})
    q = result.quality
    assert q["tables"]["transactions"]["csv_rows"] == 6
    assert q["join"]["rows"] == 6
    assert q["join"]["matched_transactions"] == 3
    assert q["join"]["unmatched_transactions"] == 3
    assert q["join"]["orphan_identity_ids"] == 1
    assert q["join"]["transaction_dt_preserved"]
    assert q["join"]["transaction_order_preserved"]
    pd.testing.assert_frame_equal(transactions, original)
    assert q["fraud"]["fraud_rate_pct"] == pytest.approx(100 / 3)
    assert q["amount"]["median"] == 175
    assert q["amount"]["sum"] == 1910
    assert q["amount"]["fraud_amount"] == 1050
    assert q["time"]["backward_steps"] == 2
    assert q["missing"]["identity"]["missing_cells"] == 1
    merged_missing = result.missing_values.query("dataset == 'merged' and column == 'DeviceType'")
    assert merged_missing["missing_count"].item() == 4
    assert merged_missing["missing_pct"].item() == pytest.approx(100 * 4 / 6, abs=0.0001)
    assert set(result.analyses) == set(ANALYSES)
    for name in list(ANALYSES)[:4]:
        assert result.analyses[name]["transactions"].sum() == 6
        assert result.analyses[name]["fraud_transactions"].sum() == 2
    product = result.analyses["01_product_risk"].set_index("product")
    assert product.loc["C", "fraud_rate_pct"] == 100
    assert product.loc["C", "fraud_amount"] == 1050
    bands = result.analyses["02_amount_bands"]
    assert bands["transactions"].tolist() == [1] * 6
    weeks = result.analyses["03_time_trends"]
    assert weeks["elapsed_week"].tolist() == [0, 1, 2]
    assert weeks["transactions"].tolist() == [2, 2, 2]
    device = result.analyses["04_identity_device"].set_index("identity_device_segment")
    assert device.loc["No identity record", "transactions"] == 3
    assert device.loc["Identity present; device missing", "transactions"] == 1


def test_email_support_threshold_includes_missing_group() -> None:
    transactions = pd.DataFrame({
        "TransactionID": range(299), "TransactionDT": range(299),
        "isFraud": [1] * 20 + [0] * 80 + [1] * 10 + [0] * 90 + [1] * 99,
        "TransactionAmt": 1.0, "ProductCD": "W",
        "P_emaildomain": ["example.test"] * 100 + [None] * 100 + ["rare.test"] * 99,
    })
    identity = pd.DataFrame({"TransactionID": pd.Series(dtype="int64"), "DeviceType": pd.Series(dtype="str")})
    result = analyze_frames(transactions, identity)
    domains = result.analyses["05_email_domains"].set_index("purchaser_email_domain")
    assert set(domains.index) == {"example.test", "(missing)"}
    assert domains.loc["example.test", "fraud_rate_pct"] == 20
    assert domains.loc["(missing)", "fraud_rate_pct"] == 10
    assert result.quality["join"]["matched_transactions"] == 0


@pytest.mark.parametrize("table", ["transactions", "identity"])
def test_rejects_duplicate_join_keys(frames, table) -> None:
    transactions, identity = frames
    if table == "transactions":
        transactions = pd.concat([transactions, transactions.iloc[[0]]], ignore_index=True)
    else:
        identity = pd.concat([identity, identity.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match=f"{table}: .*duplicate-key rows"):
        analyze_frames(transactions, identity)


def test_exact_duplicate_counts_exclude_conflicting_same_key_rows(frames) -> None:
    transactions, _ = frames
    conflicting = transactions.iloc[[0]].copy()
    conflicting["TransactionAmt"] = 999
    repeated = pd.concat([transactions, transactions.iloc[[0]], conflicting], ignore_index=True)
    stats = _table_quality(repeated)
    assert stats["duplicate_key_rows"] == 3
    assert stats["duplicate_key_groups"] == 1
    assert stats["exact_duplicate_rows"] == 2
    assert stats["exact_duplicate_groups"] == 1


@pytest.mark.parametrize("table", ["transactions", "identity"])
def test_rejects_null_join_keys(frames, table) -> None:
    transactions, identity = frames
    frame = transactions if table == "transactions" else identity
    frame["TransactionID"] = frame["TransactionID"].astype("Int64")
    frame.loc[0, "TransactionID"] = pd.NA
    with pytest.raises(ValueError, match=f"{table}: 1 null IDs"):
        analyze_frames(transactions, identity)


@pytest.mark.parametrize(("column", "value"), [
    ("isFraud", 2), ("isFraud", None),
    ("TransactionAmt", -1.0), ("TransactionAmt", float("inf")),
    ("TransactionAmt", float("nan")), ("TransactionAmt", "invalid"),
    ("TransactionDT", -1), ("TransactionDT", float("inf")),
    ("TransactionDT", None),
])
def test_invalid_values_fail_validation(frames, column, value) -> None:
    transactions, identity = frames
    transactions[column] = transactions[column].astype(object)
    transactions.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        analyze_frames(transactions, identity)


def test_row_count_mismatch_is_rejected(frames) -> None:
    with pytest.raises(ValueError, match="CSV and loaded row counts differ"):
        analyze_frames(*frames, source_row_counts={"transactions": 7, "identity": 4})


@pytest.mark.parametrize("backend", ["pandas", "duckdb"])
def test_pipeline_writes_only_aggregate_reports(tmp_path: Path, frames, backend) -> None:
    raw, reports = tmp_path / "raw", tmp_path / "reports"
    raw.mkdir()
    for frame, name in zip(frames, ["train_transaction.csv", "train_identity.csv"]):
        frame.to_csv(raw / name, index=False)
    before = {path.name: path.read_bytes() for path in raw.iterdir()}
    result = run_eda(raw, reports, backend=backend)
    assert result.quality["tables"]["transactions"]["csv_rows"] == 6
    assert result.quality["tables"]["identity"]["csv_rows"] == 4
    assert result.quality["backend"] == backend
    assert len(list(reports.glob("*.sql"))) == 5
    assert len(list(reports.glob("*.csv"))) == 6  # five analyses + column missingness
    assert "**PASS**" in (reports / "data_quality.md").read_text(encoding="utf-8")
    assert "five DuckDB analyses" in (reports / "sql_analyses.md").read_text(encoding="utf-8")
    for path in reports.glob("*.csv"):
        assert "TransactionID" not in pd.read_csv(path).columns
    assert before == {path.name: path.read_bytes() for path in raw.iterdir()}


@pytest.mark.parametrize("present", [None, "train_transaction.csv", "train_identity.csv"])
def test_missing_files_are_graceful_and_preserve_previous_reports(tmp_path: Path, capsys, present) -> None:
    raw, reports = tmp_path / "raw", tmp_path / "reports"
    raw.mkdir()
    reports.mkdir()
    if present:
        (raw / present).write_text("TransactionID\n1\n", encoding="utf-8")
    summary = reports / "data_quality.md"
    summary.write_text("previous run", encoding="utf-8")
    assert main(["--raw-dir", str(raw), "--output-dir", str(reports)]) == 2
    error = capsys.readouterr().err
    assert "Missing input file(s)" in error
    assert "Traceback" not in error
    assert summary.read_text(encoding="utf-8") == "previous run"


def test_loader_checks_both_files_before_loading(tmp_path: Path, monkeypatch) -> None:
    paths = DataPaths.from_raw_dir(tmp_path)
    paths.transaction.write_text("TransactionID\n1\n", encoding="utf-8")
    def should_not_load(*args, **kwargs):
        pytest.fail("Large CSV must not be read before checking for both inputs")
    monkeypatch.setattr("veyra.data.load_csv", should_not_load)
    with pytest.raises(FileNotFoundError, match="train_identity.csv"):
        load_veyra_data(paths)


def test_duplicate_csv_headers_are_rejected(tmp_path: Path, frames) -> None:
    transactions, identity = frames
    transactions.to_csv(tmp_path / "train_transaction.csv", index=False)
    (tmp_path / "train_identity.csv").write_text("TransactionID,DeviceType,DeviceType\n1,mobile,mobile\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate CSV column names"):
        run_eda(tmp_path, tmp_path / "reports")
