# Architecture

HealthBeauty360 follows a demo-first lakehouse pattern.

## Layers

1. Ingestion collects source extracts and synthetic generators into raw files.
2. Raw-to-bronze persists append-only operational snapshots.
3. Bronze-to-silver SQL models shape conformed dimensions and facts.
4. Feature engineering creates product and customer feature matrices in `data/features`.
5. Model training writes scored outputs and serialized artifacts to `data/models`.
6. Serving exposes demo APIs through FastAPI and the dashboard through Streamlit.

## Operational Cadence

- Daily pipeline: refresh synthetic inputs if required, run data quality checks, rebuild features, refresh churn, inventory, and trend outputs.
- Weekly pipeline: rerun data quality checks, rebuild features, retrain all five models, and update monitoring baselines.

## Runtime Outputs

- `data/synthetic`: generated demo source tables.
- `data/features`: saved feature matrices and metadata sidecars.
- `data/models`: model artifacts, forecasts, scores, and summaries.
- `data/reports`: data quality and pipeline summary reports.
- `data/pipeline_logs`: structured pipeline execution logs.

## Model Inventory

- `demand_forecast`: recursive weekly ridge-based SKU forecasting with seasonal fallback.
- `customer_segmentation`: KMeans clustering on RFM and behavioural features.
- `churn_prediction`: logistic regression with CRM and campaign engagement features.
- `inventory_scoring`: risk scoring plus anomaly detection for stock health.
- `trend_detection`: demand slope and spike detection for SKU and category momentum.