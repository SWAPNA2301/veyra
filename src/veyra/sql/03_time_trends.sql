-- Relative weeks from the supplied TransactionDT offset, not calendar dates.
-- First/last observed bins may be partial weeks.
SELECT CAST(FLOOR(TransactionDT / 604800.0) AS BIGINT) AS elapsed_week,
    MIN(TransactionDT) AS first_transaction_dt,
    MAX(TransactionDT) AS last_transaction_dt,
    COUNT(*) AS transactions,
    SUM(isFraud) AS fraud_transactions,
    ROUND(100.0 * AVG(isFraud), 4) AS fraud_rate_pct,
    ROUND(AVG(TransactionAmt), 2) AS mean_amount,
    ROUND(100.0 * AVG(CAST(has_identity AS INTEGER)), 4) AS identity_coverage_pct
FROM analysis_data
GROUP BY elapsed_week
ORDER BY elapsed_week;
