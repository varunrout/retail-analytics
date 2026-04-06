"""
Ingestion script for ONS Retail Sales Index (RSI).

Source: Office for National Statistics – Retail Sales Index
Fetches health & beauty retail sales data. In DEMO_MODE synthetic data is
generated that mirrors the statistical structure of the real series.

Output: ``data/raw/ons/ons_retail_sales.json`` (NDJSON)

Usage::

    python -m ingestion.ons_retail_sales
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.config import (
    DEMO_MODE,
    ONS_API_BASE,
    get_data_path,
    setup_logging,
)

logger: logging.Logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------
_CATEGORIES = [
    "Health stores",
    "Cosmetics & toiletries",
    "Pharmacies",
    "Total health & beauty",
]

# Relative weights/scale factors so sub-categories are plausible
_CATEGORY_SCALE: dict[str, float] = {
    "Health stores": 0.85,
    "Cosmetics & toiletries": 1.10,
    "Pharmacies": 0.95,
    "Total health & beauty": 1.00,
}

# Seasonal adjustment multipliers per month (index 0 = Jan)
_SEASONAL: list[float] = [
    0.92, 0.90, 0.96, 0.98, 1.01, 1.02,
    1.00, 0.99, 1.02, 1.04, 1.08, 1.14,
]


# ---------------------------------------------------------------------------
# Demo generation
# ---------------------------------------------------------------------------

def _generate_demo_rsi(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ONS RSI data for UK health & beauty retail.

    The series is anchored to base_index=100 in 2019 and includes:
    - A compound annual growth rate of ~3 % per year.
    - Realistic monthly seasonality (Christmas uplift, summer dip).
    - Small Gaussian noise.

    Args:
        seed: NumPy random seed for reproducibility.

    Returns:
        Tidy DataFrame with columns: ``period``, ``category``,
        ``sales_index``, ``yoy_change_pct``, ``mom_change_pct``.
    """
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2010-01")
    end = pd.Timestamp.now().to_period("M").to_timestamp()
    periods = pd.period_range(start=start, end=end, freq="M")

    records: list[dict] = []
    for cat in _CATEGORIES:
        scale = _CATEGORY_SCALE[cat]
        cagr = 0.03
        base_year = 2019

        index_values: list[float] = []
        for p in periods:
            years_from_base = p.year + (p.month - 1) / 12 - base_year
            trend = 100.0 * ((1 + cagr) ** years_from_base)
            seasonal_mult = _SEASONAL[p.month - 1]
            noise = rng.normal(0, 0.4)
            value = trend * seasonal_mult * scale + noise
            index_values.append(round(value, 2))

        arr = np.array(index_values)
        yoy = np.full(len(arr), np.nan)
        mom = np.full(len(arr), np.nan)
        yoy[12:] = ((arr[12:] - arr[:-12]) / arr[:-12]) * 100
        mom[1:] = ((arr[1:] - arr[:-1]) / arr[:-1]) * 100

        for i, p in enumerate(periods):
            records.append(
                {
                    "period": p.strftime("%Y-%m"),
                    "category": cat,
                    "sales_index": round(arr[i], 2),
                    "yoy_change_pct": round(yoy[i], 4) if not np.isnan(yoy[i]) else None,
                    "mom_change_pct": round(mom[i], 4) if not np.isnan(mom[i]) else None,
                }
            )

    df = pd.DataFrame(records)
    logger.info(
        "Generated %d synthetic ONS RSI records (%d periods × %d categories)",
        len(df),
        len(periods),
        len(_CATEGORIES),
    )
    return df


# ---------------------------------------------------------------------------
# Real API fetch (best-effort; falls back to demo)
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def _try_fetch_ons() -> pd.DataFrame | None:
    """Attempt to retrieve data from the ONS BETA API.

    Returns:
        DataFrame on success, ``None`` on any error.
    """
    url = f"{ONS_API_BASE}/datasets/retail-sales-index/editions/time-series/versions/1/observations"
    params = {"time": "*", "aggregate": "cpih1dim1A0", "geography": "K02000001"}
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("observations", [])
        if not obs:
            return None
        df = pd.json_normalize(obs)
        return df
    except Exception as exc:
        logger.warning("ONS API fetch failed: %s – falling back to demo data", exc)
        return None


def fetch_ons_data() -> pd.DataFrame:
    """Fetch ONS Retail Sales Index, falling back to synthetic demo data.

    Returns:
        Tidy DataFrame with RSI records.
    """
    if DEMO_MODE:
        logger.info("DEMO_MODE: generating synthetic ONS RSI data")
        return _generate_demo_rsi()

    result = _try_fetch_ons()
    if result is None:
        logger.warning("Real ONS data unavailable – using demo data instead")
        return _generate_demo_rsi()
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_ons_data(df: pd.DataFrame) -> Path:
    """Save ONS RSI data to NDJSON.

    Args:
        df: Tidy ONS RSI DataFrame.

    Returns:
        :class:`~pathlib.Path` of the written file.
    """
    out_dir = get_data_path("raw/ons")
    out_path = out_dir / "ons_retail_sales.json"
    with out_path.open("w", encoding="utf-8") as fh:
        for record in df.to_dict(orient="records"):
            fh.write(json.dumps(record) + "\n")
    logger.info("Saved %d records to %s", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    """Orchestrate ONS Retail Sales Index ingestion.

    Returns:
        Tidy RSI DataFrame.
    """
    logger.info("Starting ONS Retail Sales ingestion (DEMO_MODE=%s)", DEMO_MODE)
    df = fetch_ons_data()
    save_ons_data(df)
    logger.info(
        "ONS RSI ingestion complete: %d rows, periods %s–%s",
        len(df),
        df["period"].min(),
        df["period"].max(),
    )
    return df


if __name__ == "__main__":
    run()
