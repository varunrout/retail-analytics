-- dim_calendar: Date spine 2010–2030 with UK retail attributes
-- Grain: one row per calendar date
-- Partitioned: none (small dimension)
-- SCD: Type 1 (static)

WITH date_spine AS (
  SELECT
    DATE_ADD(DATE '2010-01-01', INTERVAL n DAY) AS cal_date
  FROM
    UNNEST(GENERATE_ARRAY(0, DATE_DIFF(DATE '2030-12-31', DATE '2010-01-01', DAY))) AS n
),

bank_holidays AS (
  SELECT
    CAST(JSON_VALUE(bh, '$.date') AS DATE) AS holiday_date,
    JSON_VALUE(bh, '$.title')               AS holiday_name
  FROM
    {{ source('bronze', 'bank_holidays') }},
    UNNEST(JSON_QUERY_ARRAY(raw_json, '$.["england-and-wales"].events')) AS bh
)

SELECT
  FORMAT_DATE('%Y%m%d', d.cal_date)          AS date_key,        -- YYYYMMDD integer key
  d.cal_date                                  AS full_date,
  EXTRACT(YEAR        FROM d.cal_date)        AS year,
  EXTRACT(QUARTER     FROM d.cal_date)        AS quarter,
  EXTRACT(MONTH       FROM d.cal_date)        AS month_num,
  FORMAT_DATE('%B',   d.cal_date)             AS month_name,
  EXTRACT(WEEK        FROM d.cal_date)        AS week_of_year,
  DATE_TRUNC(d.cal_date, WEEK(MONDAY))        AS week_start_date,
  EXTRACT(DAYOFWEEK   FROM d.cal_date)        AS day_of_week,     -- 1=Sun
  FORMAT_DATE('%A',   d.cal_date)             AS day_name,
  IF(EXTRACT(DAYOFWEEK FROM d.cal_date) IN (1,7), TRUE, FALSE)  AS is_weekend,
  IF(bh.holiday_date IS NOT NULL, TRUE, FALSE) AS is_uk_bank_holiday,
  bh.holiday_name                              AS uk_bank_holiday_name,
  CASE
    WHEN EXTRACT(MONTH FROM d.cal_date) IN (12, 1, 2) THEN 'Winter'
    WHEN EXTRACT(MONTH FROM d.cal_date) IN (3, 4, 5)  THEN 'Spring'
    WHEN EXTRACT(MONTH FROM d.cal_date) IN (6, 7, 8)  THEN 'Summer'
    ELSE 'Autumn'
  END                                         AS uk_season,
  CASE
    WHEN EXTRACT(MONTH FROM d.cal_date) = 12
     AND EXTRACT(DAY   FROM d.cal_date) BETWEEN 20 AND 31 THEN TRUE
    WHEN EXTRACT(MONTH FROM d.cal_date) = 11
     AND EXTRACT(DAY   FROM d.cal_date) BETWEEN 23 AND 30 THEN TRUE  -- Black Friday window
    ELSE FALSE
  END                                         AS is_peak_trading,
  CONCAT(
    CAST(EXTRACT(YEAR FROM d.cal_date) AS STRING), '-W',
    LPAD(CAST(EXTRACT(WEEK FROM d.cal_date) AS STRING), 2, '0')
  )                                           AS iso_year_week,
  CONCAT(
    CAST(EXTRACT(YEAR FROM d.cal_date) AS STRING), '-Q',
    CAST(EXTRACT(QUARTER FROM d.cal_date) AS STRING)
  )                                           AS year_quarter

FROM date_spine d
LEFT JOIN bank_holidays bh ON d.cal_date = bh.holiday_date
