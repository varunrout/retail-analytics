-- fact_inventory_position: Daily inventory snapshot per SKU
-- Grain: one row per product_key per snapshot_date
-- Partitioned: snapshot_date (MONTH)
-- Clustered: product_key
-- Source: synthetic inventory data

WITH inv AS (
  SELECT
    snapshot_date,
    product_id,
    warehouse_id,
    stock_on_hand,
    safety_stock,
    reorder_point,
    supplier_id,
    lead_time_days,
    avg_daily_units_sold_30d
  FROM {{ source('bronze', 'inventory_snapshots') }}
),

with_keys AS (
  SELECT
    i.*,
    COALESCE(p.product_key, 'UNKNOWN')     AS product_key,
    FORMAT_DATE('%Y%m%d', i.snapshot_date) AS date_key
  FROM inv i
  LEFT JOIN {{ ref('dim_product') }} p
    ON p.product_id = i.product_id
   AND p.is_current = TRUE
)

SELECT
  {{ dbt_utils.generate_surrogate_key(['product_key', 'warehouse_id', 'snapshot_date']) }}
                                           AS inventory_snapshot_key,
  snapshot_date,
  date_key,
  product_key,
  warehouse_id,
  stock_on_hand,
  safety_stock,
  reorder_point,
  lead_time_days,
  avg_daily_units_sold_30d,
  -- Derived
  ROUND(
    SAFE_DIVIDE(stock_on_hand, NULLIF(avg_daily_units_sold_30d, 0)), 1
  )                                        AS days_cover,
  (stock_on_hand <= 0)                     AS is_stockout,
  (stock_on_hand < reorder_point)          AS is_below_reorder_point,
  (stock_on_hand > (reorder_point * 3))    AS is_overstock

FROM with_keys
