-- Risk and monetary exposure by product; retain missing product codes.
SELECT
    COALESCE(ProductCD, '(missing)') AS product,
    COUNT(*) AS transactions,
    SUM(isFraud) AS fraud_transactions,
    ROUND(100.0 * AVG(isFraud), 4) AS fraud_rate_pct,
    ROUND(SUM(TransactionAmt), 2) AS total_amount,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END), 2) AS fraud_amount,
    ROUND(100.0 * SUM(CASE WHEN isFraud = 1 THEN TransactionAmt ELSE 0 END)
        / NULLIF(SUM(TransactionAmt), 0), 4) AS fraud_amount_share_pct
FROM analysis_data
GROUP BY product
ORDER BY fraud_rate_pct DESC, product;
