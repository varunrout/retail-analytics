# HealthBeauty360 — UK Health & Beauty Retail Intelligence

> A synthetic UK health and beauty retail dataset with six analytical models
> (demand forecasting, price elasticity, customer segmentation, churn, inventory
> scoring, trend detection), a FastAPI serving layer and a Streamlit dashboard.
> It runs fully offline on generated data. No cloud account is required.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Headline result (read this first)

This is an honest, self-contained analytics project on **synthetic** data, not a
production cloud platform. What it actually delivers:

- **Demand forecast**: a recursive per-SKU Ridge model. Absolute accuracy is low
  because weekly SKU demand is intermittent (mean MAPE **0.70**, WMAPE **1.06**),
  but it is scored against a 52-week seasonal-naive baseline on the same held-out
  weeks and **beats it on ~77% of SKUs** (baseline MAPE 1.23, WMAPE 1.33). The lift
  over naive is the result worth reporting, not the raw MAPE.
- **Price elasticity**: a log-log fixed-effects model recovers per-category
  elasticities from promotional price variation, with 95% CIs. Five of seven
  categories are elastic (CI upper bound below -1); each gets a margin-protected
  discount recommendation.
- **Customer segmentation**: K-Means with the number of clusters chosen by a
  silhouette sweep (**k=4**, silhouette 0.25). Each segment carries an action.
- **Churn**: a logistic model on a **leakage-free forward-window label** (no purchase
  in the next 90 days). ROC-AUC **0.73**, beating the majority-class (0.50) and
  recency-rule (0.72) baselines on a 73% base rate, with a well-calibrated score
  (calibration MAE 0.18). The synthetic generator encodes a documented customer
  lifetime/churn process, which the model recovers, so this is a genuine result, not
  a leak. Recency alone is already strong (0.72); the model adds frequency and value signal.
- **Inventory scoring** and **trend detection** (Mann-Kendall significance test)
  round out the set. Every model output ends in a decision (reorder point,
  discount, retention action, ranked "act on this" list), not a bare number.

All figures above are produced by `python -m orchestration.run_pipeline --full` and
match the artifacts in `data/models/`. Nothing is hand-typed.

### What this project is not

An earlier version of this README described a GCP/BigQuery lakehouse, a dbt project,
a Prophet + XGBoost ensemble at 8.4% MAPE and a five-router API. None of that was
in the code. Those claims have been removed. The GCP/dbt/ingestion pieces exist
only as **unwired scaffolding** and are clearly marked as future work below.

## The questions it answers

On a clearly-labelled synthetic UK health and beauty dataset (500 SKUs, ~10k
customers, ~50k transaction lines over 2021-2024):

1. How many units of each SKU will sell over the next 8 weeks, and when should we reorder?
2. How does volume respond to price, and how deep can we safely discount per category?
3. Which commercial segment does each customer belong to, and what should we do about it?
4. Which customers are likely to lapse in the next 90 days?
5. Which SKUs need reordering now?
6. Which SKUs are trending up or down with statistical significance?

## Model results in detail

| Model | Metric | Value | Baseline | Verdict |
|---|---|---|---|---|
| Demand forecast | mean MAPE | 0.70 | seasonal-naive 1.23 | beats naive on ~77% of SKUs |
| Demand forecast | WMAPE | 1.06 | seasonal-naive 1.33 | beats naive |
| Price elasticity | categories elastic | 5 of 7 | n/a | CIs exclude -1 |
| Segmentation | silhouette @ k=4 | 0.25 | swept k=2..8 | k chosen by score |
| Churn | ROC-AUC | 0.73 | majority 0.50 / recency 0.72 | beats both baselines |
| Churn | PR-AUC | 0.86 | base rate 0.73 | above base rate |

**Recovered price elasticities** (true values are injected into the synthetic
generator and documented, so the model can be validated against a known target):

| Category | Elasticity | 95% CI | Elastic? |
|---|---|---|---|
| fragrance | -1.96 | [-2.17, -1.76] | yes |
| makeup | -1.78 | [-1.96, -1.60] | yes |
| sun_care | -1.56 | [-1.76, -1.35] | yes |
| haircare | -1.30 | [-1.51, -1.08] | yes |
| bath_body | -1.29 | [-1.48, -1.11] | yes |
| skincare | -0.93 | [-1.12, -0.73] | no |
| vitamins_supplements | -0.69 | [-0.90, -0.48] | no |

## Architecture (what actually runs)

```
synthetic/            generate SKUs, customers, campaigns, inventory, 50k transactions
    │                 (documented latent price elasticity drives promotional volume)
    ▼
features/             RFM + behavioural customer features, demand/product features
    ▼
models/               demand_forecast · price_elasticity · customer_segmentation
    │                 churn_prediction · inventory_scoring · trend_detection
    ▼
data/models/*.parquet trained artifacts + summaries (predictions, scores, elasticities)
    ▼
serving/api.py        FastAPI: serves rows from the artifacts (404 if unscored)
dashboards/           Streamlit apps reading the same artifacts
```

Orchestration lives in `orchestration/` (daily operational + weekly retraining
pipelines) with data-quality checks in `data_quality/` and drift/pipeline
monitoring in `monitoring/`.

## Quickstart (no cloud required)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 1. Generate the synthetic dataset
python -m synthetic.seed_all

# 2. Run the full pipeline (features + all six models -> data/models/*.parquet)
python -m orchestration.run_pipeline --full

# 3. Serve the artifacts
uvicorn serving.api:app --port 8080
#    GET /forecast/SKU00001   /customer/CUST00001/churn_risk   /price-elasticity

# 4. Explore in the dashboard
streamlit run dashboards/app.py

# Tests and lint
pytest -q
ruff check .
```

## Repository structure

```
synthetic/          synthetic data generators (transactions, CRM, inventory, costs)
features/           RFM, behavioural, demand and product feature builders
models/             the six models, each writing a parquet artifact + summary
orchestration/      daily and weekly pipelines + CLI (run_pipeline.py)
serving/            FastAPI app serving the trained artifacts
dashboards/         Streamlit apps (executive + technical)
monitoring/         model drift and pipeline monitoring
data_quality/       data-quality checks and freshness reports
docs/               design notes and model cards
tests/              pytest suite

# Scaffolding / future work (present but NOT wired into the pipeline):
ingestion/          real-source connectors (ONS, Open Meteo, etc.) - not called
bronze_to_silver/   dbt-style SQL medallion models - no dbt_project.yml, cannot run
raw_to_bronze/      BigQuery load-job wrappers - unused offline
infra/              Terraform for a GCP deployment - not applied
```

## Scaffolding and future work

The following are deliberately kept in the repo as a sketch of where a production
build would go, but they are **not** part of the runnable pipeline and are not
required to reproduce any result above:

- **Ingestion connectors** (`ingestion/`) for real UK sources (ONS retail sales,
  Open Meteo weather, UK bank holidays, trade data). They are standalone scripts;
  the pipeline runs on synthetic data only. Install `requirements-scaffolding.txt`
  to work on them.
- **dbt / BigQuery medallion** (`bronze_to_silver/`, `raw_to_bronze/`) written as
  Jinja SQL with `{{ source }}`/`{{ ref }}` but without a `dbt_project.yml`, so it
  cannot execute. Wiring a real dbt project is the natural next step.
- **Terraform** (`infra/`) for a GCP deployment, never applied.

One modelling direction would raise this further: replacing the recursive Ridge
forecaster with a direct multi-horizon
model (gradient boosting was tested and gave no material lift on this intermittent data).

## Stack

Python 3.10+, pandas, numpy, scikit-learn, scipy, statsmodels, pymannkendall for
the models; FastAPI + uvicorn + pydantic for serving; Streamlit + plotly for
dashboards; Faker for synthetic data; pyarrow for the parquet data layer. See
`requirements.txt`. The project is deliberately dependency-light: every library in
`requirements.txt` is imported by code that runs.

## Data note

All data is synthetic and generated locally. No real customer, sales or supplier
data is used. Latent parameters that the models are meant to recover (for example
category price elasticities and the customer churn process) are documented in the
generators so results can be validated against a known ground truth.

## Licence

MIT. See [LICENSE](LICENSE).
