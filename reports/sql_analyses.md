# IEEE-CIS: five DuckDB analyses

Generated (UTC): 2026-11-23T19:56:16.129269+00:00. Full validated left join: 590,540 transactions; baseline fraud rate **3.4990%**.

Rates are percentages of transactions within each displayed group. These are in-sample, descriptive associations, not evidence of causation or validated model performance. Amounts are dataset units. All SQL and complete aggregate results accompany this report.

## Product risk and monetary exposure

[SQL](01_product_risk.sql) · [All 5 result groups](01_product_risk.csv)

Highest observed rate: **C**, 11.6873% (8,008/68,519); lowest: W, 2.0399% (8,969/439,670).

| product | transactions | fraud_transactions | fraud_rate_pct | total_amount | fraud_amount | fraud_amount_share_pct |
| --- | --- | --- | --- | --- | --- | --- |
| C | 68519 | 8,008 | 11.6873 | 2,937,570.76 | 391,421.4 | 13.3247 |
| S | 11628 | 686 | 5.8996 | 700,813.6 | 43,416 | 6.1951 |
| H | 33024 | 1,574 | 4.7662 | 2,416,368 | 246,632 | 10.2067 |
| R | 37699 | 1,426 | 3.7826 | 6,344,975 | 348,050 | 5.4854 |
| W | 439670 | 8,969 | 2.0399 | 67,339,221.37 | 2,054,325.46 | 3.0507 |

## Transaction amount bands

[SQL](02_amount_bands.sql) · [All 6 result groups](02_amount_bands.csv)

Highest observed rate: **05: [500, 1000)**, 5.7453% (962/16,744); lowest: 06: [1000, infinity), 2.4670% (187/7,580).

| amount_band | transactions | fraud_transactions | fraud_rate_pct | mean_amount | fraud_amount |
| --- | --- | --- | --- | --- | --- |
| 01: [0, 50) | 187515 | 7,480 | 3.989 | 32.08 | 210,692.05 |
| 02: [50, 100) | 160742 | 4,618 | 2.8729 | 66 | 320,696.32 |
| 03: [100, 250) | 175923 | 5,198 | 2.9547 | 142.28 | 782,372.73 |
| 04: [250, 500) | 42036 | 2,218 | 5.2764 | 338.3 | 761,360.79 |
| 05: [500, 1000) | 16744 | 962 | 5.7453 | 659.75 | 699,792.61 |
| 06: [1000, infinity) | 7580 | 187 | 2.467 | 1,690.97 | 308,930.36 |

## Relative-week fraud and coverage trends

[SQL](03_time_trends.sql) · [All 27 result groups](03_time_trends.csv)

Highest observed rate: **16.0**, 5.0612% (1,063/21,003); lowest: 3.0, 1.8523% (690/37,251).
Weeks are floor(TransactionDT / 604800), not calendar weeks. First/last bins may be partial. Preview shows the first/last three observed bins.

| elapsed_week | first_transaction_dt | last_transaction_dt | transactions | fraud_transactions | fraud_rate_pct | mean_amount | identity_coverage_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 86400 | 604792 | 23810 | 668 | 2.8055 | 129.36 | 25.7077 |
| 1 | 604802 | 1209594 | 27803 | 727 | 2.6148 | 127.71 | 39.449 |
| 2 | 1209610 | 1814393 | 34470 | 879 | 2.55 | 131.42 | 48.5495 |
| 24 | 14515200 | 15119990 | 20206 | 741 | 3.6672 | 135.16 | 23.4633 |
| 25 | 15120005 | 15724796 | 17681 | 736 | 4.1627 | 138.27 | 18.6754 |
| 26 | 15724823 | 15811131 | 2754 | 119 | 4.321 | 171.46 | 18.3733 |

## Identity coverage and device type

[SQL](04_identity_device.sql) · [All 4 result groups](04_identity_device.csv)

Highest observed rate: **Identity present; mobile**, 10.1662% (5,657/55,645); lowest: No identity record, 2.0939% (9,345/446,307).
A matched record with missing DeviceType is distinct from no identity record.

| identity_device_segment | transactions | fraud_transactions | fraud_rate_pct | mean_amount |
| --- | --- | --- | --- | --- |
| No identity record | 446307 | 9,345 | 2.0939 | 151.66 |
| Identity present; desktop | 85165 | 5,554 | 6.5215 | 92.6 |
| Identity present; mobile | 55645 | 5,657 | 10.1662 | 69.5 |
| Identity present; device missing | 3423 | 107 | 3.1259 | 86.9 |

## Purchaser-email domains (at least 100 transactions)

[SQL](05_email_domains.sql) · [All 52 result groups](05_email_domains.csv)

Highest observed rate: **mail.com**, 18.9624% (106/559); lowest: windstream.net, 0.0000% (0/305).
Minimum support: 100 transactions per domain (including a missing group). Retained 590,107 of 590,540 transactions; excluded 433 in smaller groups. Preview is the ten highest observed rates, not a risk ranking for deployment.

| purchaser_email_domain | transactions | fraud_transactions | fraud_rate_pct | mean_amount |
| --- | --- | --- | --- | --- |
| mail.com | 559 | 106 | 18.9624 | 157.61 |
| outlook.es | 438 | 57 | 13.0137 | 41.09 |
| aim.com | 315 | 40 | 12.6984 | 128.74 |
| outlook.com | 5096 | 482 | 9.4584 | 112.94 |
| hotmail.es | 305 | 20 | 6.5574 | 43.06 |
| live.com.mx | 749 | 41 | 5.474 | 44.87 |
| hotmail.com | 45250 | 2,396 | 5.295 | 99.93 |
| gmail.com | 228355 | 9,943 | 4.3542 | 128.63 |
| yahoo.fr | 143 | 5 | 3.4965 | 47.43 |
| embarqmail.com | 260 | 9 | 3.4615 | 126.77 |
