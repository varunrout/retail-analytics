# Architecture

HealthBeauty360 runs end to end on synthetic data. The live path is:
synthetic generation, feature engineering, model training, artifact persistence,
serving via FastAPI and Streamlit. A lakehouse/dbt/GCP path exists only as unwired
scaffolding (see the README "Scaffolding / future work" section).

## Live layers

1. Synthetic generation writes source tables to `data/synthetic`.
2. Feature engineering creates product and customer feature matrices in `data/features`.
3. Model training writes scored outputs and serialised artifacts to `data/models`.
4. Serving exposes the trained artifacts through FastAPI, and dashboards through Streamlit.

## Scaffolding (present but not wired into the pipeline)

- `ingestion/`: connectors for real UK sources, run standalone, not called by the pipeline.
- `raw_to_bronze/`, `bronze_to_silver/`: dbt-style SQL medallion models without a
  `dbt_project.yml`, so they cannot execute yet.
- `infra/`: Terraform for a GCP deployment, never applied.

## Operational cadence

- Daily pipeline: refresh synthetic inputs if required, run data quality checks, rebuild
  features, refresh inventory and trend outputs.
- Weekly pipeline: rerun data quality checks, rebuild features, retrain all six models
  (demand, price elasticity, segmentation, churn, inventory, trend), update monitoring baselines.

## Runtime outputs

- `data/synthetic`: generated source tables.
- `data/features`: saved feature matrices and metadata sidecars.
- `data/models`: model artifacts, forecasts, scores, elasticities, and summaries.
- `data/reports`: data quality and pipeline summary reports.
- `data/pipeline_logs`: structured pipeline execution logs.

## Model inventory

- `demand_forecast`: recursive weekly Ridge SKU forecasting, seasonal-naive baseline and fallback.
- `price_elasticity`: log-log fixed-effects category elasticity with confidence intervals.
- `customer_segmentation`: KMeans on RFM and behavioural features, silhouette-selected k.
- `churn_prediction`: logistic regression on a leakage-free forward-window label.
- `inventory_scoring`: risk scoring plus anomaly detection for stock health.
- `trend_detection`: Mann-Kendall significance test for SKU and category momentum.
