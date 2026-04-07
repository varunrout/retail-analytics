# Data Dictionary

## transactions.parquet

- `invoice_id`: normalized invoice identifier.
- `stock_code`: SKU identifier.
- `description`: product description.
- `quantity`: line quantity; negative values represent returns.
- `invoice_date`: transaction timestamp.
- `unit_price_gbp`: selling price per unit.
- `customer_id`: nullable customer identifier.
- `category_l1`: normalized product category.
- `channel`: order channel.
- `net_revenue_gbp`: revenue net of discount.
- `gross_margin_gbp`: estimated gross margin amount.

## inventory.parquet

- `stock_code`: SKU identifier.
- `stock_on_hand`: current inventory units.
- `safety_stock`: calculated buffer stock.
- `reorder_point`: stock threshold to trigger replenishment.
- `days_cover`: current days of supply.
- `lead_time_days`: supplier lead time.
- `abc_class`: value classification.

## customers.parquet

- `customer_id`: customer identifier.
- `customer_lifetime_value_gbp`: synthetic CLV.
- `total_orders`: historical order count.
- `is_active`: recent purchasing flag.
- `loyalty_tier`: Bronze, Silver, Gold, or Platinum.
- `preferred_category`: dominant category affinity.

## campaigns.parquet

- `campaign_id`: campaign identifier.
- `campaign_type`: email, sms, or push.
- `sent_date`: campaign send date.
- `open_flag`: open event.
- `click_flag`: click event.
- `conversion_flag`: conversion event.
- `revenue_attributed_gbp`: attributed revenue value.

## costs.parquet

- `stock_code`: SKU identifier.
- `landed_cost_gbp`: landed cost per unit.
- `total_cost_gbp`: total estimated unit cost.
- `gross_margin_pct`: gross margin percent.
- `amazon_fee_gbp`: reference marketplace fee.