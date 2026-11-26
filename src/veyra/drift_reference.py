"""Explicit one-time creation of an aggregate training-only drift reference."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from .data import DataPaths, JOIN_KEY, TIME_COLUMN
from .drift import POLICY, PROFILE_FILE, digest, feature_profile
from .features import ALLOWED_FEATURES
from .training_data import _header, _quoted


def training_frame(raw_dir, split, selected):
    """Shared CSV tokenization only; exclusively train rows reach Python.

    No label column is projected. Identity rows are restricted to training IDs.
    The development loader is deliberately not used: it loads validation too.
    """
    paths = DataPaths.from_raw_dir(raw_dir)
    stamps = [(p.stat().st_size, p.stat().st_mtime_ns) for p in (paths.transaction, paths.identity)]
    if stamps[0][0] != split["source_bytes"]:
        raise ValueError("Transaction source size differs from the frozen split")
    windows = [item for item in split["splits"] if item["split"] == "train"]
    if len(windows) != 1 or not split.get("integrity") or not all(split["integrity"].values()):
        raise ValueError("A successful frozen training split is required")
    train = windows[0]
    if not 0 <= train["TransactionDT_min"] <= train["TransactionDT_max"]:
        raise ValueError("Invalid training bounds")
    tx_header, id_header = _header(paths.transaction), _header(paths.identity)
    tx = [name for name in selected if name in tx_header]
    ids = [name for name in selected if name in id_header]
    if set(tx) & set(ids) or set(tx + ids) != set(selected):
        raise ValueError("Source feature schema does not match training")
    read = "header=true, delim=',', all_varchar=true, ignore_errors=false"
    with duckdb.connect(config={"threads": 4, "memory_limit": "1GB"}) as db:
        db.execute("SET enable_progress_bar = false")
        db.execute(f"CREATE TEMP TABLE training_keys AS SELECT TransactionID, CAST(TransactionDT AS DOUBLE) AS TransactionDT "
                   f"FROM read_csv(?, {read}) WHERE CAST(TransactionDT AS DOUBLE) BETWEEN ? AND ?",
                   [str(paths.transaction), train["TransactionDT_min"], train["TransactionDT_max"]])
        audit = db.execute("SELECT COUNT(*), COUNT(DISTINCT TransactionID), MIN(TransactionDT), MAX(TransactionDT) FROM training_keys").fetchone()
        if audit != (train["rows"], train["rows"], train["TransactionDT_min"], train["TransactionDT_max"]):
            raise ValueError("Training membership/count/bounds failed frozen audit")
        transaction = db.execute(
            f"SELECT k.TransactionID, k.TransactionDT, {', '.join('t.' + _quoted(n) for n in tx)} "
            f"FROM read_csv(?, {read}) t JOIN training_keys k USING (TransactionID) ORDER BY k.TransactionDT, k.TransactionID",
            [str(paths.transaction)]).df()
        identity = db.execute(f"SELECT {', '.join(_quoted(n) for n in [JOIN_KEY, *ids])} FROM read_csv(?, {read}) "
                              "WHERE TransactionID IN (SELECT TransactionID FROM training_keys)", [str(paths.identity)]).df()
    merged = transaction.merge(identity, on=JOIN_KEY, how="left", sort=False, validate="one_to_one")
    if len(merged) != train["rows"] or not merged[JOIN_KEY].equals(transaction[JOIN_KEY]):
        raise ValueError("Training join changed membership or order")
    if stamps != [(p.stat().st_size, p.stat().st_mtime_ns) for p in (paths.transaction, paths.identity)]:
        raise ValueError("Source changed during profiling")
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(merged, index=False).values.tobytes()).hexdigest()
    return merged[selected], {key: train[key] for key in ("rows", "TransactionDT_min", "TransactionDT_max")}, fingerprint


def create_reference(root: Path):
    destination = root / "reports" / PROFILE_FILE
    if destination.exists():
        raise FileExistsError("Fixed reference already exists; refusing to overwrite it")
    metrics_bytes = (root / "reports" / "training_metrics.json").read_bytes()
    split_bytes = (root / "reports" / "chronological_split_summary.json").read_bytes()
    metrics, split = json.loads(metrics_bytes), json.loads(split_bytes)
    selected = metrics["features"]["selected"]
    if (metrics.get("preprocessing_fit_split") != "train" or not selected or len(set(selected)) != len(selected)
            or any(name not in ALLOWED_FEATURES for name in selected)
            or metrics["data_provenance"]["split_audit_sha256"] != hashlib.sha256(split_bytes).hexdigest()):
        raise ValueError("Training manifest and frozen split do not agree")
    frame, window, fingerprint = training_frame(root / "data" / "raw", split, selected)
    if len(frame) < POLICY["min_reference_rows"] or len(frame) != metrics["features"]["preprocessor_fit_rows"]:
        raise ValueError("Insufficient or mismatched training reference rows")
    profile = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_split": "train", "reference_rows": len(frame), "train_window": window,
        "training_metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "split_audit_sha256": hashlib.sha256(split_bytes).hexdigest(), "model_run_id": metrics["run_id"],
        "training_features_sha256": fingerprint, "policy": POLICY,
        "method": "Train-only quantiles with explicit underflow/overflow bins; up to 20 frequent training categories (count >= 50), other/unseen and missing bins. PSI = sum((q-p)*ln(q/p)); add 1e-6 to bin proportions then renormalize. Missingness is measured before imputation. No labels, validation or test rows profiled.",
        "features": [feature_profile(name, frame[name]) for name in selected],
    }
    profile["profile_id"] = digest(profile)
    # Exclusive creation is intentional: no HTTP/background creation or updates.
    with destination.open("x", encoding="utf-8") as output:
        json.dump(profile, output, indent=2, allow_nan=False)
        output.write("\n")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        profile = create_reference(args.root)
    except Exception:
        print("Reference not created. Ensure the original training CSVs and matching frozen split/training reports exist, and no fixed drift reference already exists.")
        return 2
    print(f"Created fixed training reference: {profile['reference_rows']:,} rows, {len(profile['features'])} features. No uploaded rows, labels, validation/test profiles, model fitting or baseline updates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
