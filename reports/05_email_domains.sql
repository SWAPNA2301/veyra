-- Exclude groups with fewer than 100 transactions; missing remains a group.
-- These are descriptive associations, not causal or predictive claims.
SELECT COALESCE(P_emaildomain, '(missing)') AS purchaser_email_domain,
    COUNT(*) AS transactions,
    SUM(isFraud) AS fraud_transactions,
    ROUND(100.0 * AVG(isFraud), 4) AS fraud_rate_pct,
    ROUND(AVG(TransactionAmt), 2) AS mean_amount
FROM analysis_data
GROUP BY purchaser_email_domain
HAVING COUNT(*) >= 100
ORDER BY fraud_rate_pct DESC, transactions DESC, purchaser_email_domain;
