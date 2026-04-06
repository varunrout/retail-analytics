-- fact_weather: Daily weather observations for UK retail cities
-- Grain: one row per location per date
-- Partitioned: weather_date (MONTH)
-- Source: Open-Meteo archive API (free, no auth)

SELECT
  {{ dbt_utils.generate_surrogate_key(['weather_date', 'location']) }}
                              AS weather_key,
  CAST(weather_date AS DATE)  AS weather_date,
  FORMAT_DATE('%Y%m%d', CAST(weather_date AS DATE)) AS date_key,
  location,
  ROUND(temp_max_c, 1)        AS temp_max_c,
  ROUND(temp_min_c, 1)        AS temp_min_c,
  ROUND((temp_max_c + temp_min_c) / 2, 1) AS temp_avg_c,
  ROUND(precipitation_mm, 2)  AS precipitation_mm,
  weather_code,
  -- WMO weather interpretation
  CASE
    WHEN weather_code = 0                     THEN 'Clear sky'
    WHEN weather_code BETWEEN 1 AND 3         THEN 'Partly cloudy'
    WHEN weather_code BETWEEN 51 AND 67       THEN 'Rainy'
    WHEN weather_code BETWEEN 71 AND 77       THEN 'Snow'
    WHEN weather_code BETWEEN 80 AND 82       THEN 'Rain showers'
    WHEN weather_code BETWEEN 95 AND 99       THEN 'Thunderstorm'
    ELSE 'Overcast'
  END                         AS weather_description,
  -- Retail-relevant derived fields
  (temp_max_c >= 20)          AS is_warm_day,
  (precipitation_mm > 5)      AS is_rainy_day,
  -- Heating/cooling degree days for demand modelling
  GREATEST(0, 18 - (temp_max_c + temp_min_c) / 2) AS heating_degree_days,
  GREATEST(0, (temp_max_c + temp_min_c) / 2 - 22) AS cooling_degree_days,
  CURRENT_TIMESTAMP()         AS _dbt_loaded_at

FROM {{ source('bronze', 'weather') }}
