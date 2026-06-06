-- =====================================================================
-- analysis_queries.sql  —  Smart-meter analytics in SQL (MySQL 8)
-- Mirrors analysis.py so SQL and Python results can be cross-checked.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. SYSTEM KPIs — total demand, peak demand, load factor
-- ---------------------------------------------------------------------
WITH sys AS (
    SELECT `timestamp`, SUM(kwh) AS system_kw
    FROM readings GROUP BY `timestamp`)
SELECT
    ROUND(SUM(system_kw)/1000, 1)                AS total_mwh,
    ROUND(MAX(system_kw), 1)                      AS peak_demand_kw,
    ROUND(AVG(system_kw), 1)                      AS avg_demand_kw,
    ROUND(AVG(system_kw) / MAX(system_kw), 3)     AS load_factor
FROM sys;

-- ---------------------------------------------------------------------
-- 2. AVERAGE LOAD PROFILE by hour of day (where the system peaks)
-- ---------------------------------------------------------------------
SELECT HOUR(`timestamp`) AS hour_of_day,
       ROUND(AVG(kwh), 3) AS avg_kwh_per_meter
FROM readings
GROUP BY HOUR(`timestamp`)
ORDER BY hour_of_day;

-- ---------------------------------------------------------------------
-- 3. CONSUMPTION by customer type (join to meters)
-- ---------------------------------------------------------------------
SELECT m.customer_type,
       COUNT(DISTINCT m.meter_id)        AS meters,
       ROUND(SUM(r.kwh)/1000, 1)         AS total_mwh,
       ROUND(AVG(r.kwh), 3)              AS avg_kwh
FROM readings r JOIN meters m ON m.meter_id = r.meter_id
GROUP BY m.customer_type
ORDER BY total_mwh DESC;

-- ---------------------------------------------------------------------
-- 4. MONTHLY DEMAND (seasonal shape)
-- ---------------------------------------------------------------------
SELECT MONTH(`timestamp`) AS month,
       ROUND(SUM(kwh)/1000, 1) AS total_mwh
FROM readings
GROUP BY MONTH(`timestamp`)
ORDER BY month;

-- ---------------------------------------------------------------------
-- 5. DAILY DEMAND with 7-day moving average (window function)
--    -- the series fed into the Holt-Winters forecast
-- ---------------------------------------------------------------------
WITH daily AS (
    SELECT DATE(`timestamp`) AS d, SUM(kwh)/1000 AS mwh
    FROM readings GROUP BY DATE(`timestamp`))
SELECT d, ROUND(mwh, 1) AS mwh,
       ROUND(AVG(mwh) OVER (ORDER BY d ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) AS mwh_7d_avg
FROM daily
ORDER BY d;

-- ---------------------------------------------------------------------
-- 6. ANOMALY candidates — meters with zero-reading dropouts
-- ---------------------------------------------------------------------
SELECT meter_id,
       COUNT(*)                                              AS readings,
       SUM(CASE WHEN kwh = 0 THEN 1 ELSE 0 END)              AS zero_readings,
       ROUND(SUM(CASE WHEN kwh = 0 THEN 1 ELSE 0 END)/COUNT(*), 4) AS zero_share,
       ROUND(MAX(kwh), 2)                                    AS max_kwh
FROM readings
GROUP BY meter_id
HAVING zero_share > 0.005 OR max_kwh > 60
ORDER BY zero_share DESC;
