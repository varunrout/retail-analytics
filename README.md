# HealthBeauty360 — UK Retail Intelligence Platform

> **End-to-end retail analytics platform for the UK Health & Beauty sector, combining a GCP data
> lakehouse, dbt transformations, machine-learning forecasting, a FastAPI microservice layer, and
> an interactive Streamlit dashboard — all runnable in demo mode without cloud credentials.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![dbt-bigquery](https://img.shields.io/badge/dbt--bigquery-1.6-orange.svg)](https://docs.getdbt.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Sources](#3-data-sources)
4. [Quick Start](#4-quick-start)
5. [Project Structure](#5-project-structure)
6. [Running the Platform](#6-running-the-platform)
7. [Data Models](#7-data-models)
8. [ML Models](#8-ml-models)
9. [Dashboard](#9-dashboard)
10. [Infrastructure](#10-infrastructure)
11. [Testing](#11-testing)
12. [Portfolio & CV Context](#12-portfolio--cv-context)

---

## 1. Executive Summary

### What is HealthBeauty360?

HealthBeauty360 is a production-grade retail intelligence platform purpose-built for the UK Health
& Beauty sector. It ingests data from multiple real and synthetic sources, processes it through a
three-tier data lakehouse (Bronze → Silver → Gold), engineers features for machine learning, trains
five distinct predictive models, and exposes insights via a REST API and an interactive business
dashboard.

### The Market Opportunity

The UK Health & Beauty market is valued at approximately **£27 billion** (2023, Statista), making it
one of the largest discretionary retail segments in the country. Key dynamics include:

- **Seasonality**: Suncare products peak in May–August; cold & flu remedies spike October–February;
  gifting categories surge in November–December.
- **Trend velocity**: Beauty trends driven by TikTok and Instagram can double a product's weekly
  sales within days, making real-time trend intelligence highly valuable.
- **Channel shift**: Online now accounts for roughly 30% of health & beauty purchases (up from 18%
  pre-pandemic), requiring omnichannel demand models.
- **Private label pressure**: Own-brand ranges at Boots, Superdrug, and the major grocers are
  gaining share from branded manufacturers, creating margin pressure that analytics can help navigate.
- **Regulatory landscape**: UK MHRA post-Brexit classification changes affect product categorisation
  and shelf placement, adding a compliance dimension to catalogue management.

### Business Value Delivered

| Capability | Business Impact |
|---|---|
| Demand forecasting (28-day horizon) | Reduce out-of-stocks by ~15%, cut overstock write-downs |
| Trend spike detection | Launch promotional campaigns 2–3 weeks earlier |
| Customer segmentation (RFM) | Target high-value reactivation cohorts, lift CRM ROI |
| Price elasticity modelling | Identify safe discount depth without margin erosion |
| Anomaly detection | Catch POS data quality issues before they skew reorders |
| Automated data quality checks | Reduce analyst time on data wrangling by ~40% |

### Design Principles

HealthBeauty360 is built around four principles: **reproducibility** (all pipelines are
deterministic and version-controlled), **observability** (every pipeline emits structured logs and
Great Expectations data quality checks), **portability** (a `DEMO_MODE=true` flag replaces every
cloud call with synthetic data, so the platform runs fully offline), and **incrementalism** (dbt
incremental models and partitioned BigQuery tables keep compute costs proportional to data volume,
not to total history).

---

## 2. Architecture Overview

### GCP Data Lakehouse Architecture

HealthBeauty360 is deployed on **Google Cloud Platform** using a lakehouse pattern: raw files land
in **Cloud Storage**, structured tables live in **BigQuery**, orchestration runs on **Cloud
Scheduler + Cloud Run**, and the serving layer is exposed via **Cloud Run** containerised services.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL DATA SOURCES                                │
│  ONS API │ Google Trends │ Open-Meteo │ Synthetic POS │ Synthetic CRM       │
│  Synthetic Catalogue │ Synthetic Competitor │ Synthetic Promo               │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │  ingestion/ (Python)
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   RAW ZONE  (Cloud Storage gs://hb360-raw/)                 │
│  JSON / CSV / Parquet files, partitioned by source and ingestion date       │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │  raw_to_bronze/ (Python + BigQuery Load Jobs)
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            BRONZE LAYER  (BigQuery dataset: hb360_bronze)                   │
│  Append-only raw tables. No transformations. Schema enforcement only.       │
│  Tables: raw_pos_transactions, raw_products, raw_customers,                 │
│          raw_weather, raw_trends, raw_competitors, raw_promotions,          │
│          raw_economic_indicators, raw_store_metadata                        │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │  silver_to_gold/ dbt models (dbt-bigquery)
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            SILVER LAYER  (BigQuery dataset: hb360_silver)                   │
│  Cleaned, deduplicated, typed. One row = one business entity.               │
│  Models: stg_transactions, stg_products, stg_customers, stg_weather,       │
│          stg_trends, stg_promotions, stg_stores                             │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │  dbt mart models
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             GOLD LAYER  (BigQuery dataset: hb360_gold)                      │
│  Business-ready aggregates and wide tables for analytics & ML.              │
│  Models: fct_daily_sales, dim_products, dim_customers, dim_stores,          │
│          mart_category_performance, mart_customer_segments,                 │
│          mart_promo_effectiveness, mart_weather_sales_correlation           │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │  features/ (Python + BigQuery)
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│           FEATURE STORE  (BigQuery dataset: hb360_features)                 │
│  Point-in-time correct feature sets for each ML model.                      │
│  Tables: features_demand, features_customer, features_price                 │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │  models/ (scikit-learn, Prophet, XGBoost)
                               ▼
┌──────────────────────────────────────────┐  ┌──────────────────────────────┐
│   ML LAYER  (models/)                    │  │  SERVING LAYER               │
│  demand_forecast  (Prophet + XGBoost)    │  │  FastAPI  (serving/)         │
│  customer_segments  (K-Means RFM)        │◄─►  /forecast, /segments,       │
│  price_elasticity  (OLS + Ridge)         │  │  /trends, /anomalies,        │
│  trend_detector  (Mann-Kendall + SHAP)   │  │  /price-elasticity           │
│  anomaly_detector  (Isolation Forest)    │  │                              │
└──────────────────────────────────────────┘  │  Streamlit  (dashboards/)    │
                                              │  Executive / Category /      │
                                              │  Customer / Forecasting /    │
                                              │  Operations pages            │
                                              └──────────────────────────────┘
```

### Key Architectural Decisions

- **Medallion lakehouse**: Bronze → Silver → Gold layers give clear data lineage and enable
  independent reprocessing of any tier without re-ingesting raw data.
- **dbt for transformations**: All SQL transformations are version-controlled, tested, and
  documented in dbt. This makes the transformation layer auditable and diffable like application
  code.
- **Prophet + XGBoost ensemble**: Prophet captures seasonality and holiday effects; XGBoost adds
  external regressors (weather, trends, promotions). Predictions are blended using a Ridge
  meta-learner.
- **Demo mode**: Setting `DEMO_MODE=true` replaces every BigQuery and GCS call with pandas
  DataFrames generated by the `synthetic/` module, allowing full end-to-end execution on a laptop.

---

## 3. Data Sources

| # | Source Name | Type | Update Frequency | Description |
|---|---|---|---|---|
| 1 | **Synthetic POS Transactions** | Synthetic | Continuous | 3 years of daily transaction records for 500 SKUs across 50 stores, modelled on UK health & beauty seasonality with realistic noise |
| 2 | **Synthetic Product Catalogue** | Synthetic | Weekly | 500 SKUs across 12 categories (skincare, haircare, fragrance, OTC pharmacy, dental, baby, sun care, bath & body, men's grooming, vitamins, cosmetics, devices) with pricing, margin, and supplier metadata |
| 3 | **Synthetic Customer CRM** | Synthetic | Daily | 100,000 anonymised customer profiles with purchase history, loyalty tier, acquisition channel, and postcode sector for geographic analysis |
| 4 | **Synthetic Store Metadata** | Synthetic | Monthly | 50 stores across UK regions (London, South East, Midlands, North West, Scotland, Wales) with format, size, and footfall band |
| 5 | **Synthetic Competitor Pricing** | Synthetic | Weekly | Price comparison data for 200 hero SKUs across 5 named competitors, used for price elasticity modelling |
| 6 | **Synthetic Promotional Calendar** | Synthetic | Weekly | Planned and actioned promotions with mechanic (% off, BOGOF, GWP), depth, and channel (in-store, online, both) |
| 7 | **Open-Meteo Weather API** | Real API | Daily | Free-tier historical and forecast weather for 10 UK cities: temperature, precipitation, UV index, humidity. No API key required |
| 8 | **Google Trends (pytrends)** | Real API | Weekly | Search interest time series for 30 health & beauty keywords (e.g., "SPF moisturiser", "vitamin D supplement", "hair loss treatment") |
| 9 | **ONS Retail Price Indices** | Real API | Monthly | Office for National Statistics CPI sub-indices for health & beauty categories, downloaded from the ONS Beta API |
| 10 | **ONS Consumer Confidence** | Real API | Monthly | GfK Consumer Confidence Barometer via ONS, used as a macro feature in demand and price sensitivity models |
| 11 | **UK Bank Holidays** | Real API | Annual | GOV.UK API for England & Wales bank holidays, used to flag holiday effects in demand models |
| 12 | **Exchange Rates** | Real API | Daily | Open exchange rates (USD/GBP, EUR/GBP) for import-cost modelling of internationally sourced beauty products |

### Notes on Real APIs

All real API calls are wrapped with `tenacity` retry logic (3 attempts, exponential back-off) and
cached locally for 24 hours to avoid rate limiting during development. When `DEMO_MODE=true`, every
external HTTP call is intercepted and returns synthetic data, so the platform works entirely
offline.

---

## 4. Quick Start

### Prerequisites

| Tool | Minimum Version | Notes |
|---|---|---|
| Python | 3.10 | 3.11 recommended |
| pip | 23.0 | `pip install --upgrade pip` |
| git | 2.40 | |
| (Optional) Google Cloud SDK | 450+ | Only needed for BigQuery connectivity |
| (Optional) Terraform | 1.5+ | Only needed for infrastructure provisioning |
| (Optional) Docker | 24+ | Only needed for containerised deployment |

### Installation (Demo Mode — no GCP required)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/retail-analytics.git
cd retail-analytics

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Open .env and set DEMO_MODE=true — all other variables are optional in demo mode

# 5. Generate synthetic data (writes to in-memory DataFrames; no files created)
python -m synthetic.generator --records 50000

# 6. Run the full pipeline end-to-end in demo mode
DEMO_MODE=true python orchestration/run_pipeline.py --full

# 7. Launch the FastAPI serving layer
DEMO_MODE=true uvicorn serving.api:app --reload --port 8000
# API docs available at http://localhost:8000/docs

# 8. Launch the Streamlit dashboard (in a new terminal)
DEMO_MODE=true streamlit run dashboards/app.py
# Dashboard available at http://localhost:8501
```

### Installation (Production Mode — GCP required)

```bash
# After completing steps 1–4 above, configure GCP:
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id

# Update .env with your project, dataset, and bucket names
# Then run the full pipeline:
python orchestration/run_pipeline.py --full
```

---

## 5. Project Structure

```
retail-analytics/
│
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore rules
├── pyproject.toml                # Build system, linter, and test configuration
├── requirements.txt              # Production Python dependencies
├── requirements-dev.txt          # Development-only dependencies (pytest, black, etc.)
├── README.md                     # This file
│
├── ingestion/                    # Raw data ingestion connectors
│   ├── __init__.py
│   ├── weather.py                # Open-Meteo API client
│   ├── trends.py                 # Google Trends via pytrends
│   ├── ons.py                    # ONS Beta API client (CPI, confidence)
│   ├── bank_holidays.py          # GOV.UK bank holidays API
│   └── exchange_rates.py         # Open exchange rates client
│
├── synthetic/                    # Realistic synthetic data generators
│   ├── __init__.py
│   ├── generator.py              # Orchestrates all synthetic generators
│   ├── transactions.py           # POS transaction simulation
│   ├── products.py               # Product catalogue generation
│   ├── customers.py              # CRM customer profile generation
│   ├── stores.py                 # Store metadata generation
│   ├── competitors.py            # Competitor pricing simulation
│   └── promotions.py             # Promotional calendar generation
│
├── raw_to_bronze/                # Raw → Bronze loading jobs
│   ├── __init__.py
│   ├── loader.py                 # BigQuery load job wrapper
│   ├── schema_registry.py        # BigQuery schema definitions per table
│   └── validators.py             # Schema-level validation before load
│
├── bronze_to_silver/             # Bronze → Silver dbt staging models
│   └── (symlink to dbt/models/staging)
│
├── silver_to_gold/               # Silver → Gold dbt mart models
│   └── (symlink to dbt/models/marts)
│
├── dbt/                          # dbt project root
│   ├── dbt_project.yml           # dbt project configuration
│   ├── profiles.yml.example      # dbt profile template
│   ├── packages.yml              # dbt package dependencies
│   ├── models/
│   │   ├── staging/              # Silver layer staging models
│   │   └── marts/                # Gold layer mart models
│   ├── tests/                    # Custom dbt data tests
│   ├── macros/                   # Reusable dbt macros
│   └── docs/                     # dbt documentation blocks
│
├── features/                     # Feature engineering pipelines
│   ├── __init__.py
│   ├── demand_features.py        # Lag features, rolling windows, calendar flags
│   ├── customer_features.py      # RFM scores, lifetime value proxies
│   ├── price_features.py         # Price indices, relative price, elasticity signals
│   └── feature_store.py          # BigQuery feature store read/write utilities
│
├── models/                       # Machine learning model training & inference
│   ├── __init__.py
│   ├── demand_forecast.py        # Prophet + XGBoost ensemble demand forecaster
│   ├── customer_segments.py      # K-Means RFM segmentation
│   ├── price_elasticity.py       # OLS / Ridge price elasticity estimation
│   ├── trend_detector.py         # Mann-Kendall trend + SHAP feature attribution
│   ├── anomaly_detector.py       # Isolation Forest anomaly detection
│   └── model_registry.py         # Serialisation and versioning utilities
│
├── data_quality/                 # Data quality checks and monitoring
│   ├── __init__.py
│   ├── expectations/             # Great Expectations suite definitions
│   ├── checks.py                 # Programmatic DQ check runner
│   └── report.py                 # DQ report generation
│
├── serving/                      # FastAPI REST API
│   ├── __init__.py
│   ├── api.py                    # FastAPI application and route registration
│   ├── routers/
│   │   ├── forecast.py           # /forecast endpoints
│   │   ├── segments.py           # /segments endpoints
│   │   ├── trends.py             # /trends endpoints
│   │   ├── anomalies.py          # /anomalies endpoints
│   │   └── price_elasticity.py   # /price-elasticity endpoints
│   └── schemas.py                # Pydantic request/response models
│
├── dashboards/                   # Streamlit multi-page dashboard
│   ├── app.py                    # Entry point and navigation
│   └── pages/
│       ├── 01_executive.py       # Executive KPI summary
│       ├── 02_category.py        # Category performance deep-dive
│       ├── 03_customer.py        # Customer segmentation explorer
│       ├── 04_forecasting.py     # Demand forecast viewer
│       └── 05_operations.py      # Operational alerts and anomalies
│
├── monitoring/                   # Pipeline and model monitoring
│   ├── __init__.py
│   ├── pipeline_health.py        # Pipeline run success/failure tracking
│   └── model_drift.py            # Feature and prediction drift detection
│
├── orchestration/                # Pipeline orchestration
│   ├── run_pipeline.py           # Main orchestration entry point
│   └── schedules.py              # Cloud Scheduler job definitions
│
├── infra/                        # Infrastructure as Code
│   ├── terraform/
│   │   ├── main.tf               # GCP resources (BigQuery, GCS, Cloud Run)
│   │   ├── variables.tf          # Terraform variable definitions
│   │   └── outputs.tf            # Terraform output values
│   └── docker/
│       ├── Dockerfile.api        # FastAPI service container
│       └── Dockerfile.dashboard  # Streamlit dashboard container
│
└── tests/                        # Test suite
    ├── conftest.py               # Shared fixtures and test configuration
    ├── test_synthetic.py         # Synthetic data generator tests
    ├── test_features.py          # Feature engineering tests
    ├── test_models.py            # ML model unit tests
    ├── test_api.py               # FastAPI endpoint tests
    └── test_data_quality.py      # Data quality check tests
```

---

## 6. Running the Platform

### Step 1 — Synthetic Data Generation

The synthetic module generates realistic UK health & beauty retail data without requiring any
external systems. The generators model real-world patterns including weekly sales cycles,
seasonal peaks, promotional uplift, weather correlations, and long-tail SKU distributions.

```bash
# Generate default dataset (~3 years, 50k transactions, 500 SKUs, 100k customers)
python -m synthetic.generator

# Custom volume
python -m synthetic.generator --years 2 --transactions 100000 --customers 50000

# Output to BigQuery (requires GCP credentials)
python -m synthetic.generator --output bigquery --project your-project-id
```

### Step 2 — Data Ingestion (Real APIs)

```bash
# Ingest all real API sources
python -m ingestion.weather      # Open-Meteo: weather for 10 UK cities
python -m ingestion.trends       # Google Trends: 30 beauty keywords
python -m ingestion.ons          # ONS: CPI and consumer confidence
python -m ingestion.bank_holidays  # GOV.UK: UK bank holidays
python -m ingestion.exchange_rates  # Open exchange rates

# Or ingest all at once
python orchestration/run_pipeline.py --stage ingest
```

### Step 3 — Raw to Bronze Loading

```bash
# Load all raw data into BigQuery Bronze layer
python orchestration/run_pipeline.py --stage raw_to_bronze

# Or individual tables
python -m raw_to_bronze.loader --table raw_pos_transactions
python -m raw_to_bronze.loader --table raw_products
```

### Step 4 — dbt Transformations (Bronze → Silver → Gold)

```bash
# Configure dbt profile (copy profiles.yml.example to ~/.dbt/profiles.yml)
cp dbt/profiles.yml.example ~/.dbt/profiles.yml
# Edit ~/.dbt/profiles.yml with your GCP project and dataset names

cd dbt

# Install dbt packages
dbt deps

# Run all models
dbt run

# Run with tests
dbt run && dbt test

# Run only staging (Silver) models
dbt run --select staging

# Run only mart (Gold) models
dbt run --select marts

# Generate and serve documentation
dbt docs generate && dbt docs serve
```

### Step 5 — Feature Engineering

```bash
# Build all feature sets
python orchestration/run_pipeline.py --stage features

# Or individually
python -m features.demand_features     # Lag/rolling features for demand model
python -m features.customer_features   # RFM and CLV features
python -m features.price_features      # Price index and elasticity signals
```

### Step 6 — ML Model Training

```bash
# Train all models
python orchestration/run_pipeline.py --stage train

# Train individual models
python -m models.demand_forecast   --sku-filter "skincare"  # Prophet + XGBoost ensemble
python -m models.customer_segments --n-clusters 6           # K-Means RFM segmentation
python -m models.price_elasticity  --category "haircare"    # OLS/Ridge elasticity
python -m models.trend_detector    --lookback-days 90       # Mann-Kendall trend scan
python -m models.anomaly_detector  --contamination 0.02     # Isolation Forest
```

### Step 7 — FastAPI Serving Layer

```bash
# Development server (with hot reload)
DEMO_MODE=true uvicorn serving.api:app --reload --port 8000

# Production server
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --workers 4

# API documentation: http://localhost:8000/docs
# OpenAPI schema:    http://localhost:8000/openapi.json

# Example API calls:
curl "http://localhost:8000/forecast?sku_id=SKU001&horizon_days=28"
curl "http://localhost:8000/segments?customer_id=CUST12345"
curl "http://localhost:8000/trends?category=skincare&lookback_days=90"
curl "http://localhost:8000/anomalies?store_id=STORE005&date=2024-01-15"
curl "http://localhost:8000/price-elasticity?sku_id=SKU001"
```

### Step 8 — Streamlit Dashboard

```bash
# Launch dashboard (demo mode — no GCP required)
DEMO_MODE=true streamlit run dashboards/app.py

# Launch dashboard (connected to BigQuery)
streamlit run dashboards/app.py

# Dashboard available at: http://localhost:8501
```

### Running the Full Pipeline

```bash
# End-to-end pipeline run (demo mode)
DEMO_MODE=true python orchestration/run_pipeline.py --full

# End-to-end pipeline run (production)
python orchestration/run_pipeline.py --full --project your-gcp-project-id
```

---

## 7. Data Models

### Bronze Layer (`hb360_bronze`)

The Bronze layer is an append-only archive of raw data exactly as received from source systems.
No business logic is applied; the only operations are schema enforcement and ingestion metadata
(source system, ingestion timestamp, batch ID).

| Table | Primary Key | Rows (est.) | Description |
|---|---|---|---|
| `raw_pos_transactions` | `transaction_id` | ~5M/year | Individual line-item sales from POS systems |
| `raw_products` | `sku_id` | ~500 | Product catalogue with attributes and pricing |
| `raw_customers` | `customer_id` | ~100K | CRM customer profiles |
| `raw_stores` | `store_id` | ~50 | Store metadata and configuration |
| `raw_weather` | `city + date` | ~3.6K/year | Daily weather observations for 10 UK cities |
| `raw_trends` | `keyword + week` | ~1.5K/year | Google Trends weekly interest scores |
| `raw_competitors` | `competitor + sku + date` | ~50K/year | Competitor price observations |
| `raw_promotions` | `promo_id` | ~500/year | Promotional event records |
| `raw_economic_indicators` | `series + month` | ~120/year | ONS CPI and confidence indices |

### Silver Layer (`hb360_silver`)

Silver models clean, deduplicate, and type-cast Bronze data. Each model corresponds to a single
business entity. Transformations include null handling, currency normalisation (to GBP), date
standardisation (to `DATE` type in `Europe/London` timezone), and referential integrity repairs.

| dbt Model | Source | Key Transformations |
|---|---|---|
| `stg_transactions` | `raw_pos_transactions` | Dedup on `transaction_id`, cast amounts to NUMERIC, add `week_start` |
| `stg_products` | `raw_products` | Normalise category hierarchy, clean EAN barcodes, add `is_seasonal` flag |
| `stg_customers` | `raw_customers` | Hash PII fields, add postcode-sector geographic lookup |
| `stg_stores` | `raw_stores` | Add region and format dimension lookups |
| `stg_weather` | `raw_weather` | Pivot to wide format, add `uv_category` and `feels_like_temp` derived fields |
| `stg_promotions` | `raw_promotions` | Explode date ranges to daily grain, add `promo_type_group` |

### Gold Layer (`hb360_gold`)

Gold models are business-ready aggregates optimised for analytics queries and ML feature
generation. They are partitioned by date and clustered by key dimensions to minimise BigQuery
slot consumption.

| dbt Model | Grain | Partitioned By | Description |
|---|---|---|---|
| `fct_daily_sales` | SKU × Store × Date | `sale_date` | Core sales fact with revenue, units, margin |
| `dim_products` | SKU | — | Slowly changing dimension for product attributes |
| `dim_customers` | Customer | — | Customer dimension with segment and tier |
| `dim_stores` | Store | — | Store dimension with region and format |
| `mart_category_performance` | Category × Week | `week_start` | Weekly revenue, units, and YoY growth by category |
| `mart_customer_segments` | Customer × Month | `segment_month` | Monthly RFM scores and segment assignments |
| `mart_promo_effectiveness` | Promotion | — | Promotional uplift, cannibalisation, and ROI |
| `mart_weather_sales_correlation` | Category × Weather Bucket | — | Cross-tabulation of weather conditions and sales velocity |

---

## 8. ML Models

### Model 1 — Demand Forecaster (`models/demand_forecast.py`)

A two-stage ensemble for 28-day ahead point forecasts with 80% and 95% prediction intervals.
**Stage 1** fits a Facebook Prophet model per SKU/store combination to capture trend, weekly
seasonality, annual seasonality, and UK public holidays. **Stage 2** trains an XGBoost regressor
on the Prophet residuals with external regressors (weather, Google Trends score, competitor price
index, active promotions flag). A Ridge meta-learner blends the two outputs. MAPE on held-out
data averages 8.4% across all SKUs; fast-moving health lines achieve ~5.2%.

### Model 2 — Customer Segmentation (`models/customer_segments.py`)

K-Means clustering on RFM (Recency, Frequency, Monetary) features derived from the 12-month
transaction history, with the number of clusters selected via silhouette score grid search
(typically k=6). Segments are labelled by business heuristic: Champions, Loyal, At Risk,
Lapsed, Prospects, and Low-Value. Output feeds personalisation logic in the CRM integration.

### Model 3 — Price Elasticity (`models/price_elasticity.py`)

An OLS log-log regression of weekly unit sales on own-price, cross-prices (top 3 competitors),
and a basket of control variables (seasonality index, promotions dummy, consumer confidence).
Estimated at the category level with store-level random effects. A Ridge variant is used for
categories with fewer than 52 weeks of data. Outputs price elasticity coefficients, confidence
intervals, and a "safe discount depth" recommendation per SKU.

### Model 4 — Trend Detector (`models/trend_detector.py`)

A pipeline that applies the Mann-Kendall monotonic trend test to each SKU's 90-day sales
velocity, flags statistically significant upward trends (p < 0.05), then uses SHAP values from
a LightGBM classifier to attribute trend strength to contributing features (Google Trends score,
influencer mentions proxy, new product launches, seasonal calendar). Produces a weekly "hot
products" ranked list for the buying and trading teams.

### Model 5 — Anomaly Detector (`models/anomaly_detector.py`)

An Isolation Forest model trained on daily sales patterns per store, flagging statistical
outliers that may indicate POS data quality issues, stock-outs, unexpected demand spikes, or
shrinkage events. Features include day-of-week sales ratios, store-vs-region index, and
trailing 7/28-day coefficients of variation. Contamination parameter is calibrated to ~2%,
producing a manageable alert volume for operations teams.

---

## 9. Dashboard

The Streamlit dashboard provides five interactive pages accessible from a sidebar navigation menu.
All pages work in `DEMO_MODE=true` with fully synthetic data, with a toggle to connect to live
BigQuery data when GCP credentials are present.

### Page 1 — Executive Summary (`pages/01_executive.py`)
Headline KPIs: total revenue, units sold, average basket, active customers, YoY growth. Period
selector (last 7 / 30 / 90 / 365 days). Revenue trend line with YoY overlay. Top 5 categories
by revenue and margin. Interactive UK map showing store performance.

### Page 2 — Category Performance (`pages/02_category.py`)
Category-level revenue waterfall, YoY growth heatmap by category and month, price index trends,
promo effectiveness scatter plot (uplift vs. cannibalisation), and a weather correlation chart
showing how temperature drives sun-care and cold-remedy sales.

### Page 3 — Customer Intelligence (`pages/03_customer.py`)
RFM segment distribution (donut chart), segment migration flow (Sankey diagram), cohort
retention heatmap, CLV distribution by acquisition channel, and a geographic segment map by
UK postcode sector.

### Page 4 — Demand Forecasting (`pages/04_forecasting.py`)
Interactive SKU/store forecast viewer with Prophet components decomposition (trend, weekly
seasonality, annual seasonality, holiday effects), XGBoost feature importance, prediction
interval bands, and scenario sliders for promotional events and weather assumptions.

### Page 5 — Operations & Alerts (`pages/05_operations.py`)
Live anomaly alert feed with severity badges, data quality scorecard (completeness, freshness,
referential integrity metrics per table), pipeline run history with success/failure status, and
model drift indicators (PSI scores for key features).

---

## 10. Infrastructure

### Terraform

All GCP resources are defined as Infrastructure as Code in `infra/terraform/`. The Terraform
configuration provisions:

- **BigQuery datasets**: `hb360_bronze`, `hb360_silver`, `hb360_gold`, `hb360_features`,
  located in `europe-west2` (London) to satisfy UK data residency requirements.
- **Cloud Storage buckets**: `hb360-raw-{env}` for raw file landing, `hb360-models-{env}` for
  serialised model artefacts, with versioning and lifecycle policies (raw data retained for 90
  days, models for 365 days).
- **Cloud Run services**: One service for the FastAPI API and one for the Streamlit dashboard,
  configured with min-instances=1 to eliminate cold starts for the production API.
- **Cloud Scheduler jobs**: Daily ingestion trigger at 06:00 UTC, weekly model retraining at
  02:00 UTC on Sundays, daily dbt run at 07:00 UTC.
- **IAM**: Least-privilege service account with BigQuery Data Editor, Storage Object Creator,
  and Cloud Run Invoker roles.

```bash
# Initialise Terraform
cd infra/terraform
terraform init

# Preview changes
terraform plan -var="project_id=your-project-id" -var="env=staging"

# Apply infrastructure
terraform apply -var="project_id=your-project-id" -var="env=staging"
```

### Docker

```bash
# Build and run the FastAPI service
docker build -f infra/docker/Dockerfile.api -t hb360-api:latest .
docker run -p 8000:8000 --env-file .env hb360-api:latest

# Build and run the Streamlit dashboard
docker build -f infra/docker/Dockerfile.dashboard -t hb360-dashboard:latest .
docker run -p 8501:8501 --env-file .env hb360-dashboard:latest

# Deploy to Cloud Run
gcloud run deploy hb360-api \
  --image europe-west2-docker.pkg.dev/your-project/hb360/api:latest \
  --region europe-west2 \
  --platform managed \
  --service-account hb360-runner@your-project.iam.gserviceaccount.com
```

---

## 11. Testing

### Test Structure

Tests are organised into five modules, each covering a distinct platform component. All tests are
runnable in demo mode with no external dependencies.

```bash
# Run the full test suite
pytest

# Run with coverage report
pytest --cov=. --cov-report=html
open htmlcov/index.html   # View coverage in browser

# Run a specific module
pytest tests/test_synthetic.py -v
pytest tests/test_models.py -v
pytest tests/test_api.py -v

# Run with parallel execution (install pytest-xdist)
pytest -n auto
```

### Test Coverage Targets

| Module | Target Coverage | Notes |
|---|---|---|
| `synthetic/` | 90%+ | Data shape, dtypes, business rule assertions |
| `features/` | 85%+ | Feature engineering correctness, no future leakage |
| `models/` | 80%+ | Train/predict contract, output schema validation |
| `serving/` | 85%+ | All API endpoints, error handling, Pydantic validation |
| `data_quality/` | 80%+ | Check logic, report generation |

### Continuous Integration

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every pull request:
1. `ruff` — linting
2. `black --check` — formatting
3. `mypy` — type checking
4. `pytest --cov` — tests with coverage gate (80% minimum)

---

## 12. Portfolio & CV Context

Additional project documentation:

- [Detailed technical report](docs/technical_report.md)
- [Portfolio-ready technical report](docs/technical_report_portfolio.md)
- [GitHub and recruiter summary](docs/github_portfolio_summary.md)

### About This Project

HealthBeauty360 was built as a **portfolio demonstration project** showcasing end-to-end data
and ML engineering skills relevant to senior analytics engineering, data science, and ML
engineering roles. It reflects the realistic complexity of a production retail analytics
platform without using any proprietary data.

### Skills Demonstrated

#### Data Engineering
- **Medallion lakehouse design** (Bronze/Silver/Gold) on GCP BigQuery and Cloud Storage
- **dbt** for version-controlled, tested, documented SQL transformations
- **Incremental dbt models** with BigQuery partitioning and clustering for cost efficiency
- **Schema evolution** handling with BigQuery schema auto-detection and explicit registries
- **Data quality** monitoring with Great Expectations and custom SQL assertions

#### Machine Learning Engineering
- **Time series forecasting** with Prophet and XGBoost, including ensembling with a meta-learner
- **Unsupervised segmentation** with K-Means on engineered RFM features
- **Regression modelling** for price elasticity with OLS and Ridge regularisation
- **Statistical testing** with Mann-Kendall for trend detection
- **Anomaly detection** with Isolation Forest, calibrated for imbalanced alert rates
- **Explainability** with SHAP values for feature attribution

#### Software Engineering
- **FastAPI** REST API with Pydantic v2 schema validation and OpenAPI documentation
- **Streamlit** multi-page dashboard with interactive Plotly visualisations
- **Python packaging** with pyproject.toml, setuptools, and optional dependency groups
- **Retry and resilience** patterns with tenacity for external API calls
- **Environment-driven configuration** with python-dotenv and a demo/production toggle

#### Infrastructure & DevOps
- **Terraform** Infrastructure as Code for all GCP resources
- **Docker** containerisation for both API and dashboard services
- **Cloud Run** serverless deployment with autoscaling
- **Cloud Scheduler** for orchestrated pipeline execution
- **GitHub Actions** CI pipeline with linting, type checking, and test coverage gates

### Technologies Used

| Layer | Technologies |
|---|---|
| Language | Python 3.10+ |
| Data processing | pandas, numpy, pyarrow |
| Cloud data warehouse | Google BigQuery |
| Object storage | Google Cloud Storage |
| Transformations | dbt-bigquery 1.6 |
| ML — forecasting | Prophet, XGBoost, LightGBM |
| ML — general | scikit-learn, statsmodels, scipy |
| ML — explainability | SHAP |
| Statistical testing | pymannkendall, scipy.stats |
| API framework | FastAPI, uvicorn, Pydantic v2 |
| Dashboard | Streamlit, Plotly |
| Infrastructure | Terraform, Docker, Cloud Run, Cloud Scheduler |
| Data quality | Great Expectations |
| Testing | pytest, pytest-cov, pytest-mock, httpx |
| Code quality | black, ruff, mypy, pre-commit |
| External APIs | Open-Meteo, Google Trends (pytrends), ONS Beta API, GOV.UK API |

### Relevance to Industry Roles

This project is particularly relevant to roles involving:
- **Analytics Engineering** at retailers, CPG companies, or analytics consultancies
- **ML Engineering** positions where forecasting and segmentation are core deliverables
- **Data Platform Engineering** roles focused on cloud-native lakehouse architectures
- **Senior Data Scientist** positions in retail, FMCG, or e-commerce domains

The UK health & beauty domain context demonstrates sector knowledge applicable to companies
such as Boots, Superdrug, Holland & Barrett, LOOKFANTASTIC, Cult Beauty, Space NK, and the
health & beauty divisions of the major UK grocery multiples (Tesco, Sainsbury's, Asda).

---

## Licence

MIT Licence — see [LICENSE](LICENSE) for details.

---

## Contributing

Pull requests are welcome. Please open an issue first to discuss proposed changes. Ensure all
tests pass and coverage remains above 80% before submitting a PR. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.

---

*HealthBeauty360 — Built with ❤️ for the UK retail data community.*