# IEEE-CIS data quality summary

Generated (UTC): 2026-11-23T19:56:16.129269+00:00. Full files, no sampling. Status: **PASS**.

## Rows, schema and duplicates

| dataset | rows | columns | distinct_ids | null_ids | duplicate_key_rows | duplicate_key_groups | exact_duplicate_rows | exact_duplicate_groups | csv_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transactions | 590540 | 394 | 590540 | 0 | 0 | 0 | 0 | 0 | 590540 |
| identity | 144233 | 41 | 144233 | 0 | 0 | 0 | 0 | 0 | 144233 |

CSV row counts are independently parsed by DuckDB and equal the loaded counts. Required columns and unique CSV headers pass. Duplicate-row counts include every row in a repeated group; exact duplicates include TransactionID. Unique IDs imply zero exact duplicates; otherwise full-row checks examine every duplicate-key candidate.

## Join and labels

- Validated one-to-one left join: 590,540 rows, 434 columns; no row multiplication or loss.
- Matched: 144,233 (24.4239%); unmatched: 446,307; orphan identity IDs: 0.
- Transaction order and original TransactionDT values/dtype are preserved.
- Fraud: 20,663 / 590,540 = **3.4990%**; non-fraud: 569,877; missing/invalid labels: 0/0.

## Transaction amounts and time

Amounts are in recorded dataset units; no currency assumption, rounding of input, or outlier removal.

| min | median | mean | p95 | p99 | max | sum |
| --- | --- | --- | --- | --- | --- | --- |
| 0.251 | 68.769 | 135.0272 | 445 | 1,104 | 31,937.391 | 79,738,948.735 |

- Amount checks: missing=0, non-numeric=0, non-finite=0, negative=0, zero=0.
- Fraud-labelled amount: 3,083,844.86; share of total amount: 3.8674%. This is labelled transaction value, not measured financial loss.
- TransactionDT range: 86,400–15,811,131 seconds; observed span: 181.9992 days; backward steps: 0.
- No missing, non-numeric, non-finite or negative TransactionDT values. Future splits must use time order; no split or model is created here.

## Missing values

| dataset | columns | columns_with_missing | columns_over_50_pct_missing | all_missing_columns | missing_cells |
| --- | --- | --- | --- | --- | --- |
| transactions | 394 | 374 | 174 | 0 | 95566686 |
| identity | 41 | 38 | 12 | 0 | 2104107 |
| merged | 434 | 414 | 214 | 0 | 115523073 |

Most-missing merged columns (full per-column profile in [missing_values.csv](missing_values.csv)):

| dataset | column | rows | missing_count | missing_pct |
| --- | --- | --- | --- | --- |
| merged | id_24 | 590540 | 585793 | 99.1962 |
| merged | id_25 | 590540 | 585408 | 99.131 |
| merged | id_07 | 590540 | 585385 | 99.1271 |
| merged | id_08 | 590540 | 585385 | 99.1271 |
| merged | id_21 | 590540 | 585381 | 99.1264 |
| merged | id_26 | 590540 | 585377 | 99.1257 |
| merged | id_22 | 590540 | 585371 | 99.1247 |
| merged | id_23 | 590540 | 585371 | 99.1247 |
| merged | id_27 | 590540 | 585371 | 99.1247 |
| merged | dist2 | 590540 | 552913 | 93.6284 |

Identity-table missingness describes available records; merged missingness also includes transactions without any identity match. No imputation or row dropping was performed.

## Warnings

None from the structural/value checks. Class imbalance and high missingness remain analytical considerations.

See [SQL analyses](sql_analyses.md) and [machine-readable quality results](data_quality.json).

Runtime: Python 3.14.2, Pandas 3.1.1, DuckDB 1.5.6.
