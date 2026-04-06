-- dim_product: Canonical product dimension (SCD Type 2)
-- Grain: one row per product per valid period
-- Sources: UCI Online Retail II, Open Beauty Facts, Shopify mock

WITH uci_products AS (
  SELECT DISTINCT
    StockCode                                         AS source_product_id,
    Description                                       AS raw_title,
    'uci_online_retail'                               AS source_system,
    NULL                                              AS brand_raw,
    NULL                                              AS category_raw,
    NULL                                              AS gtin,
    NULL                                              AS ingredients_text,
    MIN(_ingested_at) OVER (PARTITION BY StockCode)   AS first_seen_date
  FROM {{ source('bronze', 'uci_transactions') }}
  WHERE StockCode IS NOT NULL
    AND Description IS NOT NULL
),

beauty_products AS (
  SELECT
    JSON_VALUE(p, '$.code')                    AS source_product_id,
    JSON_VALUE(p, '$.product_name')            AS raw_title,
    'open_beauty_facts'                        AS source_system,
    JSON_VALUE(p, '$.brands')                 AS brand_raw,
    JSON_VALUE(p, '$.categories')             AS category_raw,
    JSON_VALUE(p, '$.code')                   AS gtin,
    JSON_VALUE(p, '$.ingredients_text')       AS ingredients_text,
    CAST(_ingested_at AS DATE)                AS first_seen_date
  FROM {{ source('bronze', 'beauty_products') }},
       UNNEST(JSON_QUERY_ARRAY(raw_json, '$.products')) AS p
),

shopify_products AS (
  SELECT
    CAST(product_id AS STRING)   AS source_product_id,
    title                        AS raw_title,
    'shopify'                    AS source_system,
    vendor                       AS brand_raw,
    product_type                 AS category_raw,
    NULL                         AS gtin,
    NULL                         AS ingredients_text,
    CAST(_ingested_at AS DATE)   AS first_seen_date
  FROM {{ source('bronze', 'shopify_products') }}
),

all_products AS (
  SELECT * FROM uci_products
  UNION ALL
  SELECT * FROM beauty_products
  UNION ALL
  SELECT * FROM shopify_products
),

normalised AS (
  SELECT
    {{ dbt_utils.generate_surrogate_key(['source_system', 'source_product_id']) }} AS product_key,
    source_product_id                                   AS product_id,
    source_system,
    -- Title normalisation: trim, upper-case, remove extra spaces
    TRIM(UPPER(REGEXP_REPLACE(raw_title, r'\s+', ' ')))  AS title_clean,
    -- Brand normalisation
    TRIM(UPPER(COALESCE(brand_raw, 'UNKNOWN')))           AS brand_normalised,
    -- Category L1 / L2 from raw category string
    SPLIT(COALESCE(category_raw, 'Uncategorised'), ',')[OFFSET(0)]  AS category_l1_raw,
    CASE
      WHEN REGEXP_CONTAINS(UPPER(COALESCE(category_raw, '')), r'SKIN|MOISTUR|SERUM|TONER|CLEANSER') THEN 'Skincare'
      WHEN REGEXP_CONTAINS(UPPER(COALESCE(category_raw, '')), r'HAIR|SHAMPOO|CONDITIONER|DYE') THEN 'Haircare'
      WHEN REGEXP_CONTAINS(UPPER(COALESCE(category_raw, '')), r'BATH|SHOWER|SOAP|BODY WASH') THEN 'Bath & Body'
      WHEN REGEXP_CONTAINS(UPPER(COALESCE(category_raw, '')), r'FRAGRANCE|PERFUME|EAU DE') THEN 'Fragrance'
      WHEN REGEXP_CONTAINS(UPPER(COALESCE(category_raw, '')), r'VITAMIN|SUPPLEMENT|MINERAL') THEN 'Vitamins & Supplements'
      WHEN REGEXP_CONTAINS(UPPER(COALESCE(category_raw, '')), r'ORAL|TOOTH|DENTAL|MOUTHWASH') THEN 'Oral Care'
      WHEN REGEXP_CONTAINS(UPPER(COALESCE(category_raw, '')), r'SUN|SPF|SUNSCREEN') THEN 'Sun Care'
      WHEN REGEXP_CONTAINS(UPPER(COALESCE(category_raw, '')), r'MAKEUP|MASCARA|LIPSTICK|FOUNDATION') THEN 'Colour Cosmetics'
      ELSE 'General Health & Beauty'
    END                                                   AS category_l1,
    gtin,
    ingredients_text,
    first_seen_date,
    CURRENT_DATE()                                        AS valid_from,
    DATE '9999-12-31'                                     AS valid_to,
    TRUE                                                  AS is_current,
    CURRENT_TIMESTAMP()                                   AS _dbt_updated_at
  FROM all_products
)

SELECT * FROM normalised
