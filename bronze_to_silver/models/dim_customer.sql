-- dim_customer: Customer dimension (SCD Type 2)
-- Grain: one row per customer per valid period
-- Source: synthetic CRM data

SELECT
  {{ dbt_utils.generate_surrogate_key(['customer_id']) }}
                                        AS customer_key,
  customer_id,
  postcode_area,
  age_band,
  gender,
  acquisition_channel,
  CAST(first_purchase_date AS DATE)     AS first_purchase_date,
  CAST(registration_date AS DATE)       AS registration_date,
  email_domain,
  -- Derived geography tier
  CASE
    WHEN postcode_area IN ('EC','WC','W1','SW1','SE1','E1') THEN 'Central London'
    WHEN postcode_area LIKE 'L%'  THEN 'North West'
    WHEN postcode_area LIKE 'M%'  THEN 'Greater Manchester'
    WHEN postcode_area LIKE 'B%'  THEN 'West Midlands'
    WHEN postcode_area LIKE 'LS%' THEN 'Yorkshire'
    ELSE 'Other UK'
  END                                   AS geo_region,
  CURRENT_DATE()                        AS valid_from,
  DATE '9999-12-31'                     AS valid_to,
  TRUE                                  AS is_current,
  CURRENT_TIMESTAMP()                   AS _dbt_updated_at

FROM {{ source('bronze', 'crm_customers') }}
