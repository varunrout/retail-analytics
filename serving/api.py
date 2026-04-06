"""
FastAPI serving layer for HealthBeauty360 model scores and analytics.

Endpoints:
- GET /health
- GET /forecast/{product_id}?weeks=12
- GET /customer/{customer_id}/segment
- GET /customer/{customer_id}/churn_risk
- GET /inventory/{product_id}/score
- GET /dashboard/exec_kpis?period=weekly
"""
import logging
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="HealthBeauty360 API",
    description="UK Health & Beauty Retail Intelligence Platform API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    data_freshness: dict


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
    model_accuracy_mape: Optional[float] = None


class CustomerSegmentResponse(BaseModel):
    customer_id: str
    segment_name: str
    rfm_scores: dict
    segment_description: str
    recommended_actions: list[str]


class ChurnRiskResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_band: str
    key_risk_factors: list[dict]
    recommended_retention_action: str


class InventoryScoreResponse(BaseModel):
    product_id: str
    abc_class: str
    xyz_class: str
    stockout_risk_score: float
    dead_stock_score: float
    days_cover: float
    reorder_recommended: bool
    reorder_quantity_suggestion: int


class ExecKPIsResponse(BaseModel):
    period: str
    total_revenue_gbp: float
    units_sold: int
    avg_order_value_gbp: float
    gross_margin_pct: float
    revenue_wow_change: Optional[float] = None
    revenue_yoy_change: Optional[float] = None
    top_categories: list[dict]
    top_channels: list[dict]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_DEMO_DATA: Optional[dict] = None


def _load_demo_data() -> dict:
    """Load demo data from data/synthetic/ if available, else generate minimal demo."""
    global _DEMO_DATA
    if _DEMO_DATA is not None:
        return _DEMO_DATA

    data: dict = {}
    synthetic_dir = Path("data/synthetic")

    parquet_files = {
        "transactions": "transactions.parquet",
        "inventory": "inventory.parquet",
        "customers": "customers.parquet",
    }

    for key, fname in parquet_files.items():
        fpath = synthetic_dir / fname
        if fpath.exists():
            try:
                data[key] = pd.read_parquet(fpath)
                logger.info("Loaded %s (%d rows)", fpath, len(data[key]))
            except Exception as exc:
                logger.warning("Could not load %s: %s", fpath, exc)

    _DEMO_DATA = data
    return data


def _get_demo_product_ids(n: int = 500) -> list[str]:
    return [f"SKU-{i:04d}" for i in range(1, n + 1)]


def _get_demo_customer_ids(n: int = 10000) -> list[str]:
    return [f"CUST-{i:06d}" for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    data = _load_demo_data()
    freshness: dict = {}
    for key, df in data.items():
        freshness[key] = {"rows": len(df), "loaded": True}
    if not freshness:
        freshness["demo_mode"] = {"rows": 0, "loaded": False}
    return HealthResponse(
        status="ok",
        version="0.1.0",
        timestamp=datetime.utcnow(),
        data_freshness=freshness,
    )


@app.get("/forecast/{product_id}", response_model=ForecastResponse)
async def get_forecast(
    product_id: str,
    weeks: int = Query(default=12, ge=1, le=52),
) -> ForecastResponse:
    """Get demand forecast for a product."""
    rng = np.random.default_rng(abs(hash(product_id)) % (2**31))
    base_demand = float(rng.uniform(20, 200))
    forecasts: list[ForecastPoint] = []
    start_date = date.today() + timedelta(days=1)
    for w in range(weeks):
        fc_date = start_date + timedelta(weeks=w)
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * fc_date.month / 12)
        point = base_demand * seasonal_factor * float(rng.uniform(0.9, 1.1))
        lower = max(0.0, point * 0.75)
        upper = point * 1.25
        forecasts.append(
            ForecastPoint(
                forecast_date=fc_date,
                units_forecast=round(point, 1),
                units_lower=round(lower, 1),
                units_upper=round(upper, 1),
                method_used="demo_seasonal",
            )
        )
    return ForecastResponse(
        product_id=product_id,
        forecast_weeks=weeks,
        forecasts=forecasts,
        model_accuracy_mape=round(float(rng.uniform(8, 20)), 2),
    )


_SEGMENTS = [
    {
        "name": "Champions",
        "description": "High value, recent, frequent buyers.",
        "actions": ["Reward with loyalty points", "Early access to new products"],
    },
    {
        "name": "Loyal Customers",
        "description": "Regular purchasers with good lifetime value.",
        "actions": ["Upsell premium lines", "Invite to VIP programme"],
    },
    {
        "name": "Potential Loyalists",
        "description": "Recent buyers with moderate frequency.",
        "actions": ["Nurture with targeted emails", "Offer bundle discounts"],
    },
    {
        "name": "At Risk",
        "description": "Previously active but lapsing.",
        "actions": ["Win-back campaign", "Special discount offer"],
    },
    {
        "name": "Hibernating",
        "description": "Low recency and low frequency.",
        "actions": ["Reactivation email", "Survey to understand needs"],
    },
    {
        "name": "New Customers",
        "description": "First-time purchasers.",
        "actions": ["Welcome series", "Cross-sell complementary products"],
    },
    {
        "name": "Price Sensitive",
        "description": "Buyers who respond primarily to promotions.",
        "actions": ["Promotional offers", "Price-match communication"],
    },
]


@app.get("/customer/{customer_id}/segment", response_model=CustomerSegmentResponse)
async def get_customer_segment(customer_id: str) -> CustomerSegmentResponse:
    """Get customer segment and RFM profile."""
    rng = np.random.default_rng(abs(hash(customer_id)) % (2**31))
    seg = _SEGMENTS[int(rng.integers(0, len(_SEGMENTS)))]
    r_score = int(rng.integers(1, 6))
    f_score = int(rng.integers(1, 6))
    m_score = int(rng.integers(1, 6))
    return CustomerSegmentResponse(
        customer_id=customer_id,
        segment_name=seg["name"],
        rfm_scores={"recency": r_score, "frequency": f_score, "monetary": m_score},
        segment_description=seg["description"],
        recommended_actions=seg["actions"],
    )


@app.get("/customer/{customer_id}/churn_risk", response_model=ChurnRiskResponse)
async def get_churn_risk(customer_id: str) -> ChurnRiskResponse:
    """Get churn risk score and key factors."""
    rng = np.random.default_rng(abs(hash(customer_id + "_churn")) % (2**31))
    prob = float(rng.uniform(0.0, 1.0))
    if prob >= 0.7:
        band = "High"
        action = "Immediate win-back campaign with 20% discount."
    elif prob >= 0.35:
        band = "Medium"
        action = "Send personalised re-engagement email."
    else:
        band = "Low"
        action = "Continue standard CRM nurture programme."

    factors = [
        {"factor": "days_since_last_purchase", "importance": round(float(rng.uniform(0.1, 0.5)), 3)},
        {"factor": "purchase_frequency_decline", "importance": round(float(rng.uniform(0.05, 0.35)), 3)},
        {"factor": "avg_order_value_trend", "importance": round(float(rng.uniform(0.05, 0.25)), 3)},
    ]
    factors.sort(key=lambda x: x["importance"], reverse=True)
    return ChurnRiskResponse(
        customer_id=customer_id,
        churn_probability=round(prob, 4),
        risk_band=band,
        key_risk_factors=factors,
        recommended_retention_action=action,
    )


@app.get("/inventory/{product_id}/score", response_model=InventoryScoreResponse)
async def get_inventory_score(product_id: str) -> InventoryScoreResponse:
    """Get inventory health scores for a product."""
    rng = np.random.default_rng(abs(hash(product_id + "_inv")) % (2**31))
    abc = str(rng.choice(["A", "B", "C"], p=[0.2, 0.3, 0.5]))
    xyz = str(rng.choice(["X", "Y", "Z"], p=[0.5, 0.3, 0.2]))
    stockout_risk = round(float(rng.uniform(0, 100)), 1)
    dead_stock = round(float(rng.uniform(0, 100)), 1)
    days_cover = round(float(rng.uniform(0, 120)), 1)
    reorder = bool(stockout_risk > 60 or days_cover < 14)
    reorder_qty = int(rng.integers(50, 500)) if reorder else 0
    return InventoryScoreResponse(
        product_id=product_id,
        abc_class=abc,
        xyz_class=xyz,
        stockout_risk_score=stockout_risk,
        dead_stock_score=dead_stock,
        days_cover=days_cover,
        reorder_recommended=reorder,
        reorder_quantity_suggestion=reorder_qty,
    )


_CATEGORIES = ["Skincare", "Haircare", "Fragrance", "Bath & Body", "Vitamins & Supplements", "Oral Care", "Cosmetics"]
_CHANNELS = ["Online", "In-Store", "Marketplace", "Wholesale"]


@app.get("/dashboard/exec_kpis", response_model=ExecKPIsResponse)
async def get_exec_kpis(
    period: str = Query(default="weekly", pattern="^(daily|weekly|monthly)$"),
) -> ExecKPIsResponse:
    """Get executive KPIs for dashboard."""
    rng = np.random.default_rng(42)
    multiplier = {"daily": 1, "weekly": 7, "monthly": 30}.get(period, 7)
    revenue = round(float(rng.uniform(50_000, 120_000) * multiplier), 2)
    units = int(rng.integers(2_000, 8_000) * multiplier)
    aov = round(revenue / max(units / 3, 1), 2)
    margin = round(float(rng.uniform(0.35, 0.55)), 4)
    wow = round(float(rng.uniform(-0.05, 0.15)), 4)
    yoy = round(float(rng.uniform(0.05, 0.25)), 4)

    top_cats = []
    cat_rev = sorted(
        [{"category": c, "revenue_gbp": round(float(rng.uniform(5_000, 30_000) * multiplier), 2)} for c in _CATEGORIES],
        key=lambda x: x["revenue_gbp"],
        reverse=True,
    )
    top_cats = cat_rev[:5]

    top_channels = [
        {"channel": c, "revenue_gbp": round(float(rng.uniform(10_000, 60_000) * multiplier), 2)}
        for c in _CHANNELS
    ]
    top_channels.sort(key=lambda x: x["revenue_gbp"], reverse=True)

    return ExecKPIsResponse(
        period=period,
        total_revenue_gbp=revenue,
        units_sold=units,
        avg_order_value_gbp=aov,
        gross_margin_pct=margin,
        revenue_wow_change=wow,
        revenue_yoy_change=yoy,
        top_categories=top_cats,
        top_channels=top_channels,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
