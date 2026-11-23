# Chronological train / validation / test split

Generated (UTC): 2026-11-23T20:09:51.501595+00:00. All 590,540 transaction rows checked; **integrity PASS**.

| Split | Target rows | Actual rows | Actual % | Fraud rows | Fraud rate % | TransactionDT min | TransactionDT max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 413,378 | 413,378 | 70.000000 | 14,538 | 3.5169 | 86,400 | 10,437,996 |
| validation | 88,581 | 88,581 | 15.000000 | 3,042 | 3.4341 | 10,438,003 | 13,151,840 |
| test | 88,581 | 88,581 | 15.000000 | 3,083 | 3.4804 | 13,151,880 | 15,811,131 |

## Policy and integrity

- Targets are 70/15/15 of rows, not elapsed-time duration; no random split or label stratification.
- Stable ascending TransactionDT sort. Nominal cuts are floor(0.70*n) and floor(0.85*n). Each cut uses the nearest admissible complete-timestamp gap; equal-distance gaps go earlier. At least one timestamp group is reserved for each remaining split.
- Timestamp-group adjustment required: no. Large tied groups can change the target proportions; fewer than three distinct timestamps is rejected.
- IDs are non-null, unique within and disjoint across splits; every source row appears exactly once.
- Every split is ordered. max(train.TransactionDT) < min(validation.TransactionDT), and max(validation.TransactionDT) < min(test.TransactionDT); equal timestamps never cross boundaries.
- Fraud labels are binary, non-null and unchanged per TransactionID; TransactionDT is also unchanged. Fraud prevalence is measured after membership is fixed and is not forced to match across periods.
- This audit reads only the three split-defining columns. The same splitter accepts all transaction/identity features, preserving original columns, dtypes and indices. No raw rows or ID manifests were saved.

## Downstream use

No preprocessing or model was fitted. Future imputers, encoders, scalers and feature selection must be fitted only on training data, then applied unchanged to validation and test. Time-derived or aggregate features must not use future transactions; keep the target out of features. Strict timestamp boundaries do not by themselves prove every feature was available at prediction time.

Validation is for development; preserve test as the final holdout. Test-label prevalence here is only an integrity report, not a basis for changing the split or selecting features. Earlier full-data EDA reports are descriptive, not training-fitted transformations.

[Aggregate CSV](chronological_split_summary.csv) | [Machine-readable audit](chronological_split_summary.json)
