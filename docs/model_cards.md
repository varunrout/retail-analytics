# Model Cards

All models train on synthetic data and write parquet artifacts to `data/models/`.
Metrics quoted here are produced by `python -m orchestration.run_pipeline --full`.

## demand_forecast

- Objective: predict weekly SKU demand for replenishment and planning.
- Inputs: lagged weekly sales, rolling averages, seasonal encodings.
- Method: recursive per-SKU Ridge regression, seasonal-naive fallback for short histories.
- Output: `demand_forecast_predictions.parquet`, `demand_forecast_metrics.parquet`.
- Evaluation: scored against a 52-week seasonal-naive baseline on the identical held-out
  weeks. Mean MAPE 0.66 vs baseline 1.22; WMAPE 1.06 vs 1.40; beats naive on ~77% of SKUs.
  Absolute error is high (intermittent weekly demand); the reported result is the lift over naive.
- Decision: each forecast carries a `reorder_point` (lead-time demand + safety stock).

## price_elasticity

- Objective: estimate category price elasticity of demand and safe discount depth.
- Inputs: weekly units and effective price (list price net of promotional discount).
- Method: log-log OLS with per-SKU fixed effects, so elasticity is identified from
  within-SKU promotional price moves. 95% CIs and p-values reported.
- Output: `price_elasticity.parquet`.
- Result: 4 of 7 categories elastic (CI upper bound below -1). Validated against the
  documented latent elasticities in the synthetic generator.
- Decision: promote / hold / test per category, with a margin-protected max discount.

## customer_segmentation

- Objective: group customers into commercially useful cohorts.
- Inputs: RFM metrics, basket behaviour, CRM profile fields.
- Method: standardised KMeans; k chosen by a silhouette sweep over k=2..8 (selected k=4,
  silhouette ~0.27), with ordered business labels.
- Output: `customer_segments.parquet`.
- Decision: a recommended commercial action per segment.

## churn_prediction

- Objective: estimate churn propensity for CRM actioning.
- Inputs: RFM and behavioural features computed strictly before a cut-off, CRM attributes,
  campaign engagement.
- Label: forward-window definition (no purchase in the next 90 days). This replaces the
  previous recency-derived label, which leaked because recency was also a feature.
- Method: logistic regression (class-balanced) with one-hot encoded categoricals.
- Output: `customer_churn_scores.parquet`.
- Evaluation: ROC-AUC ~0.57 vs majority-class 0.50 and recency-rule ~0.49 baselines;
  PR-AUC ~0.92 on a 90% base rate; calibration curve reported. Honest read: churn is only
  weakly predictable on this synthetic data. Each customer carries a recommended action.

## inventory_scoring

- Objective: rank inventory by stockout and dead-stock risk.
- Inputs: inventory positions, replenishment thresholds, demand volatility, cost context.
- Output: `inventory_scores.parquet`.
- Method: weighted business scoring plus Isolation Forest anomaly signal.
- Decision: `reorder_recommended` flag and a suggested reorder quantity.

## trend_detection

- Objective: identify accelerating, growing, stable, and declining SKU demand.
- Inputs: weekly demand, rolling demand statistics.
- Method: Mann-Kendall monotonic trend test per SKU (p < 0.05) combined with a recent
  z-score; accelerating/declining labels require a significant test.
- Output: `trend_detection_sku.parquet`, `trend_detection_category.parquet`.
- Decision: a ranked "act on this" priority list.
