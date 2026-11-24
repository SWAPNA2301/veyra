# Train-only ML pipelines: validation comparison

Run `20261124T144826624784Z`. Training: 413,378 rows; validation: 88,581 rows. Existing chronological windows reused unchanged. **Test holdout not evaluated or used for selection.**

| Model | PR-AUC (AP) | ROC-AUC | Precision | Recall | F1 | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_skill | 0.034341 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 85,539 | 0 | 3,042 | 0 |
| logistic_regression | 0.174703 | 0.794616 | 0.097018 | 0.678172 | 0.169752 | 66,338 | 19,201 | 979 | 2,063 |
| lightgbm | 0.235530 | 0.840563 | 0.120268 | 0.709073 | 0.205654 | 69,761 | 15,778 | 885 | 2,157 |

Confusion matrices use actual rows [non-fraud, fraud] and predicted columns [non-fraud, fraud]: `[[TN, FP], [FN, TP]]`. Threshold is fixed at 0.5 for every model; no threshold tuning. Metrics use the original, unweighted validation population.

Selected for future work by validation AP: **lightgbm**; AP lift over no-skill: 6.86x. This is a development comparison, not a held-out test estimate or deployment approval.

## Leakage controls and feature policy

- Frozen split audit bounds, row counts, label counts and time ranges are checked for train/validation. IDs are unique and disjoint, with strict chronological boundaries. The API accepts no test DataFrame.
- Shared CSVs must be scanned, but static feature projections and time filters are applied before materialization; no test features/labels are profiled, transformed, fitted, predicted or scored. CSV type inference is disabled. No full-data preprocessing or train+validation refit occurs.
- Excluded: TransactionID, isFraud, absolute TransactionDT, identifier-like card/address codes, anonymous C/D/V aggregates/deltas, and all fields outside the allowlist. These anonymous features are withheld because their past-only availability is unverified, not because leakage was proven.
- Included fields: TransactionAmt, dist1, dist2, ProductCD, card4, card6, P_emaildomain, R_emaildomain, M1, M2, M3, M4, M5, M6, M7, M8, M9, DeviceType, DeviceInfo, id_30, id_31, id_33. id_30/id_31/id_33 are device OS/browser/screen attributes, not row IDs. This conservative baseline deliberately does not use every competition feature.
- Train-only median imputation (all-empty numeric fields retained with zero fallback), missing indicators and scaling; categorical missing tokens and train-fitted rare-category grouping. Unknown categories use the infrequent bucket when available, otherwise all-zero encoding. No target encoding.
- One-hot outputs capped at 64 per categorical field; minimum frequency 20. 414 encoded features; float32 CSR training matrix: 77.27 MiB.
- Logistic Regression uses train-derived balanced class weights. LightGBM uses train negative/positive ratio 27.434310; no resampling. Weighted-model outputs are not calibrated fraud probabilities.
- LightGBM early stopping monitors validation average precision only; best iteration 255. Logistic Regression iterations: 324.
- The constant no-skill model uses the training fraud prior. Its validation AP equals validation prevalence, ROC-AUC is 0.5, and at threshold 0.5 it predicts no fraud for this imbalanced dataset.
- [Full-data weekly EDA](03_time_trends.sql) is not a training feature: its outcome rates include future periods. No weekly fraud rate or future-derived aggregate is fed to these models.

## Saved pipelines

- [no_skill](../models/20261124T144826624784Z/no_skill.joblib): fitted preprocessing + classifier; serialization predictions verified.
- [logistic_regression](../models/20261124T144826624784Z/logistic_regression.joblib): fitted preprocessing + classifier; serialization predictions verified.
- [lightgbm](../models/20261124T144826624784Z/lightgbm.joblib): fitted preprocessing + classifier; serialization predictions verified.

Model artifacts and learned vocabularies are Git-ignored. Load only trusted joblib files, using this source package and the library versions recorded in the metrics; no serving/UI is added.

[Full metrics/config/provenance](training_metrics.json) | [Metrics CSV](training_metrics.csv)

## Metric and API references

PR-AUC here is [scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), not trapezoidal PR interpolation. Category handling follows [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html); validation stopping uses [LightGBM early stopping](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html).
