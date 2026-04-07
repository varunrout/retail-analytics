# Model Cards

## demand_forecast

- Objective: predict weekly SKU demand for replenishment and planning.
- Inputs: lagged weekly sales, rolling averages, seasonal encodings.
- Output: `data/models/demand_forecast_predictions.parquet`.
- Fallback: seasonal naive forecast for sparse or short-history SKUs.

## customer_segmentation

- Objective: group customers into commercially useful cohorts.
- Inputs: RFM metrics, basket behaviour, CRM profile fields.
- Output: `data/models/customer_segments.parquet`.
- Method: standardized KMeans clustering with ordered business labels.

## churn_prediction

- Objective: estimate churn propensity for CRM actioning.
- Inputs: RFM metrics, behaviour, CRM attributes, campaign engagement.
- Output: `data/models/customer_churn_scores.parquet`.
- Method: logistic regression with one-hot encoded categorical features.

## inventory_scoring

- Objective: rank inventory by stockout and dead-stock risk.
- Inputs: inventory positions, replenishment thresholds, demand volatility, cost context.
- Output: `data/models/inventory_scores.parquet`.
- Method: weighted business scoring plus Isolation Forest anomaly signal.

## trend_detection

- Objective: identify accelerating, growing, stable, and declining SKU demand.
- Inputs: weekly demand, rolling demand statistics, relative trend slope.
- Output: `data/models/trend_detection_sku.parquet` and `data/models/trend_detection_category.parquet`.
- Method: rolling demand diagnostics plus slope and spike classification.