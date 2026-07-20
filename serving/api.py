"""FastAPI serving layer for HealthBeauty360 model scores and analytics.

Every endpoint serves rows from the artifacts written by the training pipeline
(``data/models/*.parquet``). Nothing is fabricated: if a SKU or customer has not
been scored the endpoint returns 404, and if the artifacts are missing entirely
it returns 503. The dashboard KPIs are computed from the transaction ledger.

Run the pipeline first (``python -m orchestration.run_pipeline --full``) so the
artifacts exist, then ``uvicorn serving.api:app``.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")
SYNTHETIC_DIR = Path("data/synthetic")

app = FastAPI(
    title="HealthBeauty360 API",
    description="UK Health & Beauty Retail Intelligence Platform API (serves trained artifacts)",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Artifact loading (cached; call the pipeline to refresh the parquet files)
# ---------------------------------------------------------------------------

_CACHE: dict[str, pd.DataFrame] = {}


def _load(path: Path) -> pd.DataFrame | None:
    key = str(path)
    if key not in _CACHE:
        if not path.exists():
            return None
        _CACHE[key] = pd.read_parquet(path)
    return _CACHE.get(key)


def _require(path: Path) -> pd.DataFrame:
    df = _load(path)
    if df is None:
        raise HTTPException(status_code=503, detail=f"Artifact not available: {path.name}. Run the training pipeline first.")
    return df


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    artifacts: dict


class ForecastPoint(BaseModel):
    forecast_date: date
    units_forecast: float
    units_lower: float
    units_upper: float
    method_used: str


class ForecastResponse(BaseModel):
    product_id: str
    forecast_weeks: int
    forecasts: list[ForecastPoint]
    validation_mape: float | None = None
    baseline_mape: float | None = None
    beats_baseline: bool | None = None
    reorder_point: float | None = None


class CustomerSegmentResponse(BaseModel):
    customer_id: str
    segment_name: str
    rfm_scores: dict
    recommended_action: str


class ChurnRiskResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_band: str
    recommended_action: str


class InventoryScoreResponse(BaseModel):
    product_id: str
    abc_class: str
    xyz_class: str
    stockout_risk_score: float
    dead_stock_score: float
    days_cover: float
    reorder_recommended: bool
    reorder_quantity_suggestion: int


class ElasticityResponse(BaseModel):
    category: str
    elasticity: float
    ci_low: float
    ci_high: float
    is_elastic: bool
    recommended_max_discount: float
    recommended_action: str


class ExecKPIsResponse(BaseModel):
    period: str
    total_revenue_gbp: float
    units_sold: int
    avg_order_value_gbp: float
    gross_margin_pct: float | None = None
    top_categories: list[dict]
    top_channels: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    artifacts = {
        name: (MODELS_DIR / f"{name}.parquet").exists()
        for name in [
            "demand_forecast_predictions",
            "customer_segments",
            "customer_churn_scores",
            "inventory_scores",
            "price_elasticity",
            "trend_detection_sku",
        ]
    }
    return HealthResponse(
        status="ok" if any(artifacts.values()) else "no_artifacts",
        version="1.0.0",
        timestamp=datetime.utcnow(),
        artifacts=artifacts,
    )


@app.get("/forecast/{product_id}", response_model=ForecastResponse)
async def get_forecast(product_id: str, weeks: int = Query(default=8, ge=1, le=52)) -> ForecastResponse:
    predictions = _require(MODELS_DIR / "demand_forecast_predictions.parquet")
    rows = predictions[predictions["stock_code"] == product_id].sort_values("forecast_date").head(weeks)
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No forecast for product {product_id}")

    metrics = _load(MODELS_DIR / "demand_forecast_metrics.parquet")
    meta = metrics[metrics["stock_code"] == product_id] if metrics is not None else None
    validation_mape = float(meta["validation_mape"].iat[0]) if meta is not None and not meta.empty else None
    baseline_mape = float(meta["baseline_mape"].iat[0]) if meta is not None and not meta.empty else None
    beats_baseline = bool(meta["beats_baseline"].iat[0]) if meta is not None and not meta.empty else None

    points = [
        ForecastPoint(
            forecast_date=pd.to_datetime(row["forecast_date"]).date(),
            units_forecast=float(row["forecast_units"]),
            units_lower=float(row["forecast_lower"]),
            units_upper=float(row["forecast_upper"]),
            method_used=str(row["method_used"]),
        )
        for _, row in rows.iterrows()
    ]
    return ForecastResponse(
        product_id=product_id,
        forecast_weeks=len(points),
        forecasts=points,
        validation_mape=validation_mape,
        baseline_mape=baseline_mape,
        beats_baseline=beats_baseline,
        reorder_point=float(rows["reorder_point"].iat[0]) if "reorder_point" in rows.columns else None,
    )


@app.get("/customer/{customer_id}/segment", response_model=CustomerSegmentResponse)
async def get_customer_segment(customer_id: str) -> CustomerSegmentResponse:
    segments = _require(MODELS_DIR / "customer_segments.parquet")
    row = segments[segments["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No segment for customer {customer_id}")
    row = row.iloc[0]
    return CustomerSegmentResponse(
        customer_id=customer_id,
        segment_name=str(row["segment_name"]),
        rfm_scores={
            "recency": int(row.get("r_score", 0)),
            "frequency": int(row.get("f_score", 0)),
            "monetary": int(row.get("m_score", 0)),
        },
        recommended_action=str(row.get("recommended_action", "")),
    )


@app.get("/customer/{customer_id}/churn_risk", response_model=ChurnRiskResponse)
async def get_churn_risk(customer_id: str) -> ChurnRiskResponse:
    scores = _require(MODELS_DIR / "customer_churn_scores.parquet")
    row = scores[scores["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No churn score for customer {customer_id}")
    row = row.iloc[0]
    return ChurnRiskResponse(
        customer_id=customer_id,
        churn_probability=round(float(row["churn_probability"]), 4),
        risk_band=str(row["risk_band"]),
        recommended_action=str(row.get("recommended_action", "")),
    )


@app.get("/inventory/{product_id}/score", response_model=InventoryScoreResponse)
async def get_inventory_score(product_id: str) -> InventoryScoreResponse:
    scores = _require(MODELS_DIR / "inventory_scores.parquet")
    row = scores[scores["stock_code"] == product_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No inventory score for product {product_id}")
    row = row.iloc[0]
    return InventoryScoreResponse(
        product_id=product_id,
        abc_class=str(row.get("abc_class", "")),
        xyz_class=str(row.get("xyz_class", "")),
        stockout_risk_score=round(float(row.get("stockout_risk_score", 0.0)), 2),
        dead_stock_score=round(float(row.get("dead_stock_score", 0.0)), 2),
        days_cover=round(float(row.get("days_cover", 0.0)), 1),
        reorder_recommended=bool(row.get("reorder_recommended", False)),
        reorder_quantity_suggestion=int(row.get("reorder_quantity_suggestion", 0)),
    )


@app.get("/price-elasticity", response_model=list[ElasticityResponse])
async def get_price_elasticity() -> list[ElasticityResponse]:
    table = _require(MODELS_DIR / "price_elasticity.parquet")
    return [
        ElasticityResponse(
            category=str(r["category"]),
            elasticity=float(r["elasticity"]),
            ci_low=float(r["ci_low"]),
            ci_high=float(r["ci_high"]),
            is_elastic=bool(r["is_elastic"]),
            recommended_max_discount=float(r["recommended_max_discount"]),
            recommended_action=str(r["recommended_action"]),
        )
        for _, r in table.iterrows()
    ]


@app.get("/price-elasticity/{category}", response_model=ElasticityResponse)
async def get_price_elasticity_category(category: str) -> ElasticityResponse:
    table = _require(MODELS_DIR / "price_elasticity.parquet")
    row = table[table["category"].str.lower() == category.lower()]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No elasticity estimate for category {category}")
    r = row.iloc[0]
    return ElasticityResponse(
        category=str(r["category"]),
        elasticity=float(r["elasticity"]),
        ci_low=float(r["ci_low"]),
        ci_high=float(r["ci_high"]),
        is_elastic=bool(r["is_elastic"]),
        recommended_max_discount=float(r["recommended_max_discount"]),
        recommended_action=str(r["recommended_action"]),
    )


@app.get("/dashboard/exec_kpis", response_model=ExecKPIsResponse)
async def get_exec_kpis(period: str = Query(default="weekly", pattern="^(daily|weekly|monthly)$")) -> ExecKPIsResponse:
    tx = _require(SYNTHETIC_DIR / "transactions.parquet").copy()
    tx["invoice_date"] = pd.to_datetime(tx["invoice_date"])
    span = {"daily": 1, "weekly": 7, "monthly": 30}[period]
    cutoff = tx["invoice_date"].max() - pd.Timedelta(days=span)
    window = tx[(tx["invoice_date"] > cutoff) & (tx["quantity"] > 0)]
    if window.empty:
        window = tx[tx["quantity"] > 0]

    revenue_col = "net_revenue_gbp" if "net_revenue_gbp" in window.columns else None
    revenue = float(window[revenue_col].sum()) if revenue_col else float((window["quantity"] * window["unit_price_gbp"]).sum())
    units = int(window["quantity"].sum())
    invoices = window["invoice_no"].nunique() if "invoice_no" in window.columns else max(units, 1)
    aov = round(revenue / max(invoices, 1), 2)

    gross_margin = None
    inventory = _load(SYNTHETIC_DIR / "inventory.parquet")
    if inventory is not None and revenue_col and {"sku_id", "unit_cost_gbp"}.issubset(inventory.columns):
        costs = inventory.set_index("sku_id")["unit_cost_gbp"]
        merged = window.assign(unit_cost=window["stock_code"].map(costs)).dropna(subset=["unit_cost"])
        if not merged.empty:
            cogs = float((merged["quantity"] * merged["unit_cost"]).sum())
            rev_m = float(merged[revenue_col].sum())
            gross_margin = round((rev_m - cogs) / rev_m, 4) if rev_m > 0 else None

    cat_col = "category" if "category" in window.columns else "category_l1"
    top_categories = (
        window.groupby(cat_col)[revenue_col if revenue_col else "quantity"].sum()
        .sort_values(ascending=False).head(5)
        .reset_index().rename(columns={cat_col: "category", (revenue_col or "quantity"): "revenue_gbp"})
        .to_dict("records")
    )
    top_channels = (
        window.groupby("channel")[revenue_col if revenue_col else "quantity"].sum()
        .sort_values(ascending=False)
        .reset_index().rename(columns={"channel": "channel", (revenue_col or "quantity"): "revenue_gbp"})
        .to_dict("records")
    ) if "channel" in window.columns else []

    return ExecKPIsResponse(
        period=period,
        total_revenue_gbp=round(revenue, 2),
        units_sold=units,
        avg_order_value_gbp=aov,
        gross_margin_pct=gross_margin,
        top_categories=[{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in top_categories],
        top_channels=[{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in top_channels],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
