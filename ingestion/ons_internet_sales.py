"""
Ingestion script for ONS Internet Retail Sales Index.

Source: Office for National Statistics – Internet Retail Sales Index
Covers the growth of online retail as a share of total retail in the UK.
In DEMO_MODE, synthetic data is generated showing online share growth from
~10 % in 2010 to ~28 % by 2023.

Output: ``data/raw/ons/ons_internet_sales.json`` (NDJSON)

Usage::

    python -m ingestion.ons_internet_sales
"""

import json
import logging
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
# Channel definitions
# ---------------------------------------------------------------------------
_CHANNELS = [
    "All internet retailing",
    "Non-food internet retailing",
    "Predominantly food internet retailing",
    "Health & beauty internet retailing",
]

# Base internet share in 2010 and target share by 2023 per channel
_SHARE_PARAMS: dict[str, tuple[float, float]] = {
    "All internet retailing": (10.0, 28.0),
    "Non-food internet retailing": (14.0, 36.0),
    "Predominantly food internet retailing": (2.0, 8.0),
    "Health & beauty internet retailing": (8.0, 30.0),
}

# Seasonal multipliers (index 0 = Jan) – Q4 uplift for gift/festive online buying
_SEASONAL: list[float] = [
    0.98, 0.94, 0.97, 0.99, 1.01, 1.00,
    1.00, 0.99, 1.01, 1.02, 1.03, 1.10,
]


# ---------------------------------------------------------------------------
# Demo generation
# ---------------------------------------------------------------------------

def _generate_demo_internet_sales(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ONS Internet Retail Sales Index data.

    Simulates online retail growth from ~10 % internet share in Jan 2010 to
    ~28 % by Dec 2023, with a spike during the COVID-19 period (2020-2021)
    and modest reversion afterwards.

    Args:
        seed: NumPy random seed for reproducibility.

    Returns:
        Tidy DataFrame with columns: ``period``, ``channel``,
        ``sales_index``, ``internet_share_pct``, ``yoy_change_pct``.
    """
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2010-01")
    end = pd.Timestamp.now().to_period("M").to_timestamp()
    periods = pd.period_range(start=start, end=end, freq="M")

    records: list[dict] = []
    for channel in _CHANNELS:
        share_start, share_end = _SHARE_PARAMS[channel]
        n = len(periods)

        # Build internet share trajectory (logistic-ish growth)
        t = np.linspace(0, 1, n)
        base_share = share_start + (share_end - share_start) * (3 * t**2 - 2 * t**3)

        # COVID bump: 2020-03 → 2021-06 adds ~6 % extra share
        covid_bump = np.zeros(n)
        for i, p in enumerate(periods):
            if (p.year == 2020 and p.month >= 3) or p.year == 2021:
                months_in = max(0, (p.year - 2020) * 12 + p.month - 3)
                covid_bump[i] = 6.0 * np.exp(-0.05 * months_in)

        share = base_share + covid_bump + rng.normal(0, 0.3, n)
        share = np.clip(share, 0.5, 60.0)

        # Sales index: base 100 in Jan 2019; grows faster than total retail
        base_year_idx = next(
            (i for i, p in enumerate(periods) if p.year == 2019 and p.month == 1),
            0,
        )
        index_values = np.zeros(n)
        for i, p in enumerate(periods):
            years_from_base = p.year + (p.month - 1) / 12 - 2019
            trend = 100.0 * (1.10 ** years_from_base)  # ~10% CAGR for online
            seasonal_mult = _SEASONAL[p.month - 1]
            noise = rng.normal(0, 0.5)
            if p.year == 2020 and p.month >= 3:
                trend *= 1.15  # COVID uplift
            index_values[i] = trend * seasonal_mult + noise

        yoy = np.full(n, np.nan)
        yoy[12:] = ((index_values[12:] - index_values[:-12]) / index_values[:-12]) * 100

        for i, p in enumerate(periods):
            records.append(
                {
                    "period": p.strftime("%Y-%m"),
                    "channel": channel,
                    "sales_index": round(float(index_values[i]), 2),
                    "internet_share_pct": round(float(share[i]), 2),
                    "yoy_change_pct": round(float(yoy[i]), 4) if not np.isnan(yoy[i]) else None,
                }
            )

    df = pd.DataFrame(records)
    logger.info(
        "Generated %d synthetic ONS internet sales records (%d periods × %d channels)",
        len(df),
        len(periods),
        len(_CHANNELS),
    )
    return df


# ---------------------------------------------------------------------------
# Real API fetch (best-effort; falls back to demo)
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def _try_fetch_ons_internet() -> pd.DataFrame | None:
    """Attempt to retrieve internet sales data from the ONS BETA API.

    Returns:
        DataFrame on success, ``None`` on any error.
    """
    url = (
        f"{ONS_API_BASE}/datasets/internet-retail-sales-index"
        "/editions/time-series/versions/1/observations"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("observations", [])
        if not obs:
            return None
        return pd.json_normalize(obs)
    except Exception as exc:
        logger.warning("ONS internet sales API fetch failed: %s", exc)
        return None


def fetch_ons_data() -> pd.DataFrame:
    """Fetch ONS Internet Retail Sales Index, falling back to synthetic data.

    Returns:
        Tidy DataFrame with internet sales index records.
    """
    if DEMO_MODE:
        logger.info("DEMO_MODE: generating synthetic ONS internet sales data")
        return _generate_demo_internet_sales()

    result = _try_fetch_ons_internet()
    if result is None:
        logger.warning("Real ONS internet data unavailable – using demo data")
        return _generate_demo_internet_sales()
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_ons_data(df: pd.DataFrame) -> Path:
    """Save ONS Internet Retail Sales data to NDJSON.

    Args:
        df: Tidy internet sales DataFrame.

    Returns:
        :class:`~pathlib.Path` of the written file.
    """
    out_dir = get_data_path("raw/ons")
    out_path = out_dir / "ons_internet_sales.json"
    with out_path.open("w", encoding="utf-8") as fh:
        for record in df.to_dict(orient="records"):
            fh.write(json.dumps(record) + "\n")
    logger.info("Saved %d records to %s", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    """Orchestrate ONS Internet Retail Sales ingestion.

    Returns:
        Tidy internet sales DataFrame.
    """
    logger.info("Starting ONS Internet Sales ingestion (DEMO_MODE=%s)", DEMO_MODE)
    df = fetch_ons_data()
    save_ons_data(df)
    logger.info(
        "ONS Internet Sales ingestion complete: %d rows, periods %s–%s",
        len(df),
        df["period"].min(),
        df["period"].max(),
    )
    return df


if __name__ == "__main__":
    run()
