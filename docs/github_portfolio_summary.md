# HealthBeauty360 GitHub And Recruiter Summary

## One-line summary

HealthBeauty360 is a self-contained retail analytics project on synthetic UK health and
beauty data: feature engineering, six analytical models evaluated against baselines, an
artifact-backed FastAPI service, and business/technical dashboards, all reproducible offline.

## What this project does

- generates a labelled synthetic retail dataset (500 SKUs, ~10k customers, ~50k transactions)
- validates datasets with structured data-quality checks
- builds reusable product and customer feature sets
- trains and scores six analytical model workflows, each against a baseline
- persists artifacts and reports for inspection
- serves the trained artifacts through a FastAPI service and two Streamlit dashboards

## Why it stands out

Most portfolio projects stop at notebooks or over-claim in the README. This one is honest
and end to end:

- every model is scored against a sensible baseline on held-out data, and the result is
  reported straight (the demand model beats seasonal-naive on ~77% of SKUs; churn recovers a documented
  lifetime process at ROC-AUC 0.73, beating its baselines)
- the API serves real model artifacts, not fabricated numbers
- daily and weekly orchestration with monitoring baselines
- reproducible, test-backed execution with CI (ruff + pytest)

## Implemented model workflows

1. demand forecasting (Ridge, seasonal-naive baseline, reorder-point decision)
2. price elasticity (log-log fixed effects, confidence intervals, discount guidance)
3. customer segmentation (KMeans, silhouette-selected k, per-segment action)
4. churn prediction (leakage-free forward-window label, ROC-AUC 0.73, calibrated)
5. inventory scoring (risk score plus anomaly detection)
6. trend detection (Mann-Kendall significance test, ranked action list)

## Current technical highlights

- 50,000 synthetic transactions; 500 SKU-level and ~3,200 customer feature rows
- demand: MAPE 0.70 vs seasonal-naive 1.23, beats naive on ~77% of SKUs
- elasticity: 5 of 7 categories elastic, recovered against a known target with 95% CIs
- churn: ROC-AUC 0.73 (baselines 0.50 / 0.72), calibrated (MAE 0.18), recovers a documented churn process
- CI green: ruff lint, synthetic seed, and pytest on every push

## Tech stack

Python, pandas, numpy, scikit-learn, statsmodels, pymannkendall, FastAPI, Streamlit,
Plotly, pytest, ruff. A dbt/BigQuery/Terraform path exists as unwired scaffolding.

## What this demonstrates to employers

- honest model evaluation: baselines, confidence intervals, and results reported cleanly
- ML workflow design beyond a single notebook
- orchestration, artifact management, and serving that actually reads the artifacts
- technical documentation and observability thinking

## Good resume framing

> Built a reproducible retail analytics project on synthetic UK health and beauty data:
> feature engineering, six models each evaluated against a baseline (demand forecasting,
> price elasticity, segmentation, churn, inventory scoring, trend detection), an
> artifact-backed FastAPI service, dual dashboards, and CI.

## Good interview framing

Lead with the honesty: every model is measured against a baseline. Churn recovers a
documented lifetime/churn process at ROC-AUC 0.73 (beating a strong recency baseline),
and where signal is genuinely limited the project says so. That, plus fixing a real
label-leakage bug and wiring the API to real artifacts, is the story worth telling.
