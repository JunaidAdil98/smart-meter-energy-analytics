-- =====================================================================
-- schema.sql  —  Smart-meter warehouse (MySQL 8 / standard SQL)
-- Load: schema.sql -> load meters.csv + readings.csv -> analysis_queries.sql
-- =====================================================================

DROP TABLE IF EXISTS `readings`;
DROP TABLE IF EXISTS `meters`;

CREATE TABLE `meters` (
  `meter_id`        VARCHAR(12)  NOT NULL PRIMARY KEY,
  `customer_type`   VARCHAR(16)  NOT NULL,
  `base_load_kw`    DECIMAL(8,3) NOT NULL,
  `is_faulty_truth` TINYINT      NOT NULL
);

CREATE TABLE `readings` (
  `meter_id`  VARCHAR(12)  NOT NULL,
  `timestamp` DATETIME     NOT NULL,
  `kwh`       DECIMAL(10,4) NOT NULL,
  `temp_c`    DECIMAL(5,1) NOT NULL,
  KEY `ix_meter` (`meter_id`),
  KEY `ix_ts`    (`timestamp`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
