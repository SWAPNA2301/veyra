-- Coverage is measured by a matching TransactionID, not by a non-null device.
WITH segments AS (
    SELECT *, CASE
        WHEN NOT has_identity THEN 'No identity record'
        WHEN DeviceType IS NULL THEN 'Identity present; device missing'
        ELSE 'Identity present; ' || DeviceType
    END AS identity_device_segment
    FROM analysis_data
)
SELECT identity_device_segment,
    COUNT(*) AS transactions,
    SUM(isFraud) AS fraud_transactions,
    ROUND(100.0 * AVG(isFraud), 4) AS fraud_rate_pct,
    ROUND(AVG(TransactionAmt), 2) AS mean_amount
FROM segments
GROUP BY identity_device_segment
ORDER BY transactions DESC, identity_device_segment;
