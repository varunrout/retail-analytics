-- fact_market_signal: External market signals (ONS + Google Trends)
-- Grain: one row per signal_date + signal_type + category/keyword
-- Partitioned: signal_date (MONTH)

WITH ons_signals AS (
  SELECT
    PARSE_DATE('%Y-%m', period)       AS signal_date,
    'ons_retail_sales_index'          AS signal_type,
    category                          AS dimension_value,
    sales_index                       AS signal_value,
    yoy_change_pct                    AS yoy_change
  FROM {{ source('bronze', 'ons_retail_sales') }}

  UNION ALL

  SELECT
    PARSE_DATE('%Y-%m', period)       AS signal_date,
    'ons_internet_sales_index'        AS signal_type,
    category                          AS dimension_value,
    sales_index                       AS signal_value,
    yoy_change_pct                    AS yoy_change
  FROM {{ source('bronze', 'ons_internet_sales') }}
),

trends_signals AS (
  SELECT
    PARSE_DATE('%Y-%m-%d', week_start) AS signal_date,
    'google_trends_interest'           AS signal_type,
    keyword                            AS dimension_value,
    relative_interest                  AS signal_value,
    NULL                               AS yoy_change
  FROM {{ source('bronze', 'google_trends') }}
)

SELECT
  {{ dbt_utils.generate_surrogate_key(['signal_date', 'signal_type', 'dimension_value']) }}
                    AS market_signal_key,
  signal_date,
  FORMAT_DATE('%Y%m%d', signal_date)  AS date_key,
  signal_type,
  dimension_value,
  ROUND(signal_value, 4)              AS signal_value,
  ROUND(yoy_change, 4)               AS yoy_change,
  CURRENT_TIMESTAMP()                 AS _dbt_loaded_at

FROM ons_signals
UNION ALL
SELECT
  {{ dbt_utils.generate_surrogate_key(['signal_date', 'signal_type', 'dimension_value']) }},
  signal_date,
  FORMAT_DATE('%Y%m%d', signal_date),
  signal_type,
  dimension_value,
  ROUND(signal_value, 4),
  yoy_change,
  CURRENT_TIMESTAMP()
FROM trends_signals
