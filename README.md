# Veyra

Veyra is a production-oriented starting point for fraud-detection data
science. This repository currently focuses on reliable ingestion and data
quality checks; model training and a user interface are intentionally out of
scope.

## Project layout

```text
data/raw/       Local input CSVs (ignored by Git)
notebooks/      Exploratory analysis notebooks
reports/        Generated data-quality reports
src/veyra/      Reusable loading and profiling code
tests/          Automated tests
```

The expected input files are `train_transaction.csv` and
`train_identity.csv` under `data/raw/`. Raw datasets and secrets must remain
local and must never be committed.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Data checks

```python
from pathlib import Path

from veyra import (
    DataPaths,
    fraud_class_distribution,
    load_veyra_data,
    merge_transaction_identity,
    missing_value_profile,
    profile_duplicates,
)

paths = DataPaths.from_raw_dir(Path("data/raw"))
transactions, identity = load_veyra_data(paths)
merged = merge_transaction_identity(transactions, identity)

missing = missing_value_profile(merged)
duplicates = profile_duplicates(transactions, subset=["TransactionID"])
fraud_distribution = fraud_class_distribution(transactions)
```

`TransactionDT` is required in the transaction schema and is preserved during
the identity merge for chronological train/validation/test splits.

## Reproducible full-data EDA

After installing the package, run from the project root:

```powershell
python -m veyra.eda --raw-dir data/raw --output-dir reports
```

For a source checkout without an editable installation, set `$env:PYTHONPATH = "src"`
before running that command. `--backend duckdb` selects the alternative loader.
Both loaders return full Pandas DataFrames; allow several GB of available memory.
Neither mode is a streaming or out-of-core pipeline.

The command reuses the existing loading, merge and profiling functions. It independently
counts CSV records with DuckDB, checks required columns, join-key uniqueness/nulls,
binary labels, finite nonnegative amounts/times, exact duplicates and every column's
missingness. It verifies that the left join preserves row count, order and original
`TransactionDT` values. Zero amounts and orphan identity records are reported as
warnings. It does not impute, deduplicate, trim outliers or create train/test splits.

The five packaged [SQL analyses](src/veyra/sql/) cover product risk, amount bands,
relative-week trends, identity/device coverage and purchaser-email domains. The last
uses a minimum group size of 100; descriptive rates are not model evaluation results.

Outputs (aggregate-only, safe to inspect without raw rows):

- [Data quality summary](reports/data_quality.md) and machine-readable JSON.
- [Five SQL analysis summaries](reports/sql_analyses.md), runnable SQL copies and
  full aggregate CSV results. The SQL reads `analysis_data`, which the command
  registers from the validated merged DataFrame.
- [Per-column missingness](reports/missing_values.csv) for both inputs and the join.

Missing files, malformed inputs or failed validation produce an actionable message
and exit code 2, with no successful report for that run. Existing reports remain from
their previous timestamp. Fix the inputs and rerun; no files are downloaded automatically.
All files under [data/raw/](data/raw/) except the directory placeholder are ignored by Git.

## Leakage-safe chronological splits

Use [the splitter](src/veyra/splitting.py) on transactions or the existing joined
DataFrame. It returns independent DataFrames with all original columns, dtypes
and row indices; it never fits or transforms features:

```python
from veyra.splitting import chronological_split, summarize_splits, validate_splits

splits = chronological_split(merged)
train, validation, test = splits.train, splits.validation, splits.test
validate_splits(merged, splits)
print(summarize_splits(splits))
```

Targets are **70% / 15% / 15% of rows**, with a stable ascending `TransactionDT`
sort and no label stratification. Nominal cuts are `floor(0.70*n)` and
`floor(0.85*n)`. Each cut moves to the nearest admissible gap between complete
timestamp groups (earlier gap wins an equal-distance tie), reserving a timestamp
group for each remaining split. Thus equal timestamps never cross boundaries,
and both boundaries are strictly chronological. Actual proportions can differ
when timestamps tie; large tied groups can cause large deviations. Fewer than
three distinct timestamps is rejected.

Validation rejects missing/duplicate IDs, invalid timestamps or fraud labels,
overlapping/missing IDs, unordered splits, shared timestamps across splits,
and changes to the label/time associated with an ID. Every row appears once;
labels remain binary 0/1, but fraud rates need not match between time periods.
Single-class periods are reported without resampling.

Run the real-data membership audit from the project root (use the same setup
or `PYTHONPATH` option as the EDA command):

```powershell
python -m veyra.splitting --raw-dir data/raw --output-dir reports
```

This command loads all rows but only `TransactionID`, `TransactionDT` and
`isFraud`, because features cannot influence membership. It saves the
[split summary](reports/chronological_split_summary.md) plus aggregate CSV/JSON;
it does not export raw split rows or ID manifests. Missing files and invalid
inputs fail clearly with exit code 2, leaving prior successful reports unchanged.

Fit any future imputation, encoding, scaling and feature selection **on training
data only**, then apply the fitted transformations to validation/test. Do not use
future transactions in time-dependent features or use `isFraud` as a feature.
Use validation for development and retain test as the final holdout; its fraud
rate here is only an integrity statistic. Earlier full-data EDA is descriptive,
not a training-fitted transformation. Temporal splitting alone does not establish
that every feature was available at prediction time. No preprocessing or models
are fitted by the split utility.

## Tests

```powershell
python -m pytest -q
```

The tests use small temporary fixtures and do not require reading or
committing the raw datasets.
