-- Fixed, non-overlapping bands; amounts are in the dataset's recorded units.
WITH bands AS (
    SELECT *, CASE
        WHEN TransactionAmt < 50 THEN '01: [0, 50)'
        WHEN TransactionAmt < 100 THEN '02: [50, 100)'
        WHEN TransactionAmt < 250 THEN '03: [100, 250)'
        WHEN TransactionAmt < 500 THEN '04: [250, 500)'
        WHEN TransactionAmt < 1000 THEN '05: [500, 1000)'
        ELSE '06: [1000, infinity)'
    END AS amount_band
    FROM analysis_data
)
SELECT amount_band,
    COUNT(*) AS transactions,
    SUM(isFraud) AS fraud_transactions,
    ROUND(100.0 * AVG(isFraud), 4) AS fraud_rate_pct,
    ROUND(AVG(TransactionAmt), 2) AS mean_amount,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END), 2) AS fraud_amount
FROM bands
GROUP BY amount_band
ORDER BY amount_band;
