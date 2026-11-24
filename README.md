# Veyra

Veyra is a production-oriented fraud-detection data science project with
validated ingestion, chronological holdouts and train-only ML pipelines.
User interfaces, serving and deployment remain out of scope.

## Project layout

```text
data/raw/       Local input CSVs (ignored by Git)
notebooks/      Exploratory analysis notebooks
reports/        Generated data-quality reports
models/         Local fitted pipelines and run metadata (ignored by Git)
src/veyra/      Reusable ingestion, splitting, preprocessing and training code
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

## Train-only ML comparison

After installing the dependencies, reuse the existing split audit:

```powershell
python -m veyra.training --raw-dir data/raw --split-report reports/chronological_split_summary.json --model-dir models --output-dir reports
```

As above, use `$env:PYTHONPATH = "src"` if the package is not installed editable.
The [loader](src/veyra/training_data.py) uses the saved train/validation windows;
it does not recompute splits for model selection. CSVs are scanned with explicit
schema roles and type inference disabled, but test rows are filtered out before
materialization. Only train/validation row counts, labels and boundaries are
checked. The [training API](src/veyra/training.py) accepts no test DataFrame.

The conservative [feature policy](src/veyra/features.py) excludes TransactionID,
isFraud, absolute TransactionDT, identifier-like card/address codes, anonymous
C/D/V count/delta/aggregate features with unverified past-only availability, and
all other columns outside a static allowlist. It retains transaction amount,
distance, product, card type/network, email-domain, match and device attributes.
The device OS/browser/screen attributes id_30/id_31/id_33 are not row IDs. This
baseline intentionally does not use all competition features; further feature
inclusion requires checking prediction-time availability, not inspecting test scores.

Both models share preprocessing fitted **only on train**: median imputation
with missing indicators and a zero fallback for all-empty numeric fields,
numeric scaling, categorical missing tokens, and sparse rare-category one-hot
encoding. Vocabulary/frequency grouping is train-derived, capped at 64 columns
per categorical feature, with a minimum count of 20. Unknown values go to the
rare bucket when it exists, otherwise all-zero encoding. No target encoding,
resampling or full-data feature selection is performed. Matrices are float32 CSR.

- Logistic Regression: balanced weights from training labels, fixed C=1.
- LightGBM: training negative/positive class ratio, at most 400 boosting rounds,
  validation-average-precision early stopping with patience 30, four CPU threads.
- No-skill comparator: constant probabilities equal to the training fraud prior.

Model comparison uses **validation average precision** (reported as PR-AUC),
ROC-AUC, precision, recall, F1 and confusion matrices. The classification
threshold is fixed at 0.5 for all models, without threshold search; weighted
outputs are not calibrated fraud probabilities. Validation metrics are
unweighted. The best of the two learned models is named by validation AP;
there is no train+validation refit, test evaluation, UI or deployment. The
[full-data weekly outcome analysis](reports/03_time_trends.sql) is descriptive
and is never used as a training feature.

Outputs:

- [Validation comparison](reports/training_summary.md), [metrics CSV](reports/training_metrics.csv),
  and [full metrics/config/provenance](reports/training_metrics.json).
- A unique run directory under [models/](models/) with three complete fitted
  preprocessing/classifier pipelines, SHA-256 checksums and a metrics snapshot.
  Serialization is checked against validation predictions. The folder and
  joblib files are Git-ignored; raw datasets stay local and unchanged.

To reuse a trusted local pipeline (use the artifact path in the metrics):

```python
import joblib

pipeline = joblib.load("models/<run-id>/lightgbm.joblib")
probabilities = pipeline.predict_proba(new_transactions)[:, 1]
```

Use the recorded package versions and this source package. Do not load untrusted
joblib artifacts. The existing EDA and split commands remain available unchanged.

## Validation-only threshold optimization

Reuse the saved LightGBM pipeline without retraining or changing its default
decision rule:

```powershell
python -m veyra.thresholding --raw-dir data/raw --split-report reports/chronological_split_summary.json --training-metrics reports/training_metrics.json --output-dir reports --min-recall 0.70
```

Use the same editable-install or `PYTHONPATH` setup as above. The
[threshold utility](src/veyra/thresholding.py) verifies the saved model checksum
and frozen split fingerprint, reuses the existing development loader unchanged,
discards its training frame, and calls only `predict_proba` on validation. Shared
CSV scans filter test rows before materialization. The 0.5 baseline must reproduce
the saved validation metrics before optimization can proceed. No preprocessing,
calibration or model is fitted; no test frame is created or scored.

The exact O(n log n) sweep evaluates all distinct validation scores plus 0, 0.5,
1 and a predict-none sentinel just above the maximum score. The prediction rule
is `score >= threshold`; tied scores stay together, and no-alert precision is
zero. It compares baseline 0.5 with maximum F1 and maximum precision subject to
recall >= 0.70. Ties prefer higher recall, then higher precision for max-F1,
then higher threshold. The recall constraint is never relaxed.

Outputs under [reports/](reports/):

- [Comparison and methodology](reports/threshold_optimization.md).
- [Every threshold's aggregate metrics](reports/threshold_metrics.csv),
  [selected operating points](reports/threshold_comparison.csv), and
  [exact thresholds/provenance](reports/threshold_optimization.json).
- [Metric-vs-threshold chart](reports/threshold_metrics.png),
  [precision-recall chart](reports/threshold_precision_recall.png), and
  [confusion matrices](reports/threshold_confusion_matrices.png).

Only aggregate counts and metrics are exported, not raw transactions, identifiers
or per-row predictions. Thresholds remain analysis outputs: existing training,
model files and prediction defaults are unchanged. JSON retains full-precision
thresholds; use those values rather than the report's rounded display values
when reproducing a decision. A predict-none sentinel can exceed 1 by one
floating-point step if any saved score equals 1.

The same validation data previously selected the model/early-stopping iteration
and now selects thresholds. These are development estimates, not unbiased test
results or a guarantee of recall on future data. No threshold is automatically
deployed or set as a default.

## Tests

```powershell
python -m pytest -q
```

The tests use small temporary fixtures and do not require reading or
committing the raw datasets.
