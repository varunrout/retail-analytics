-- fact_transaction: Core transaction fact table
-- Grain: one row per order line (order_id + stock_code)
-- Partitioned: invoice_date (MONTH)
-- Clustered: channel_key, product_key

WITH uci_raw AS (
  SELECT
    Invoice                                     AS order_id,
    StockCode                                   AS source_product_id,
    Description                                 AS product_description,
    Quantity                                    AS quantity,
    InvoiceDate                                 AS invoice_date,
    Price                                       AS unit_price_gbp,
    Customer_ID                                 AS source_customer_id,
    Country                                     AS customer_country,
    _ingested_at
  FROM {{ source('bronze', 'uci_transactions') }}
  WHERE Quantity > 0           -- exclude cancellations in fact (handle separately)
    AND Price > 0
    AND StockCode IS NOT NULL
),

with_keys AS (
  SELECT
    t.*,
    COALESCE(p.product_key, 'UNKNOWN')          AS product_key,
    COALESCE(c.customer_key, 'UNKNOWN')         AS customer_key,
    -- Channel assignment: UCI is predominantly web/ecommerce
    CASE
      WHEN t.customer_country = 'United Kingdom' THEN 1  -- Shopify UK
      ELSE 2                                              -- eBay UK (proxy for export)
    END                                         AS channel_key,
    FORMAT_DATE('%Y%m%d', DATE(t.invoice_date)) AS date_key
  FROM uci_raw t
  LEFT JOIN {{ ref('dim_product') }} p
    ON p.source_product_id = t.source_product_id
   AND p.source_system = 'uci_online_retail'
   AND p.is_current = TRUE
  LEFT JOIN {{ ref('dim_customer') }} c
    ON c.source_customer_id = CAST(t.source_customer_id AS STRING)
   AND c.is_current = TRUE
)

SELECT
  {{ dbt_utils.generate_surrogate_key(['order_id', 'source_product_id', 'invoice_date']) }}
                                                AS transaction_key,
  order_id,
  DATE(invoice_date)                            AS invoice_date,
  date_key,
  product_key,
  customer_key,
  channel_key,
  quantity,
  ROUND(unit_price_gbp, 2)                      AS unit_price_gbp,
  ROUND(quantity * unit_price_gbp, 2)           AS gross_sales_gbp,
  -- Estimated gross margin: 45% for health & beauty (industry benchmark)
  ROUND(quantity * unit_price_gbp * 0.45, 2)   AS gross_margin_estimate_gbp,
  customer_country,
  FALSE                                         AS is_return,
  _ingested_at

FROM with_keys
