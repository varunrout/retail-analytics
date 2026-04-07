# Feature Catalog

## Product and Demand Features

- `total_units_sold_12m`: trailing 12-month unit sales.
- `total_revenue_12m`: trailing 12-month revenue.
- `avg_unit_price`: average selling price.
- `price_cv`: price variability coefficient.
- `return_rate`: share of returns by SKU.
- `rolling_28d_units`: trailing 4-week demand.
- `rolling_28d_cv`: trailing 4-week demand variability.
- `days_cover`: inventory cover based on current demand.
- `stockout_probability`: stockout risk within supplier lead time.
- `abc_class`: value-based inventory class.
- `xyz_class`: demand variability class.

## Customer Features

- `recency_days`: days since last purchase.
- `frequency`: order count in the lookback window.
- `monetary_value`: spend in the lookback window.
- `avg_order_value`: mean order spend.
- `avg_basket_size`: mean units per order.
- `category_breadth`: distinct categories purchased.
- `days_between_orders`: average inter-order interval.
- `purchase_frequency_trend`: increasing, stable, or declining order cadence.
- `preferred_season`: highest-spend season.

## Calendar Features

- `month_sin` and `month_cos`: cyclic month encoding.
- `week_sin` and `week_cos`: cyclic week encoding.
- `is_weekend`, `quarter`, and `fiscal_quarter`: retail calendar context.