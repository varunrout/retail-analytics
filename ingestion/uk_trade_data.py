"""
UK Trade Data ingestion (HMRC Overseas Trade Statistics).

Focus: cosmetics & toiletries HS codes 3303-3307, 3401.
DEMO_MODE generates synthetic trade data based on realistic UK cosmetics
import/export patterns.

Output: ``data/raw/trade/uk_trade_data.parquet``

Usage::

    python -m ingestion.uk_trade_data
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
    HS_CODES_COSMETICS,
    get_data_path,
    setup_logging,
)

logger: logging.Logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# HS code descriptions
# ---------------------------------------------------------------------------
_HS_DESCRIPTIONS: dict[str, str] = {
    "3303": "Perfumes and toilet waters",
    "3304": "Beauty or make-up preparations and preparations for the care of the skin",
    "3305": "Preparations for use on the hair",
    "3306": "Preparations for oral or dental hygiene",
    "3307": "Pre-shave, shaving or after-shave preparations, deodorants, bath preparations",
    "3401": "Soap; organic surface-active products and preparations for use as soap",
}

# Top trade partner countries with approximate share weights
_IMPORT_PARTNERS: list[tuple[str, float]] = [
    ("France", 0.20),
    ("USA", 0.15),
    ("South Korea", 0.12),
    ("Germany", 0.10),
    ("Japan", 0.09),
    ("China", 0.12),
    ("Italy", 0.07),
    ("Netherlands", 0.06),
    ("Poland", 0.05),
    ("Belgium", 0.04),
]

_EXPORT_PARTNERS: list[tuple[str, float]] = [
    ("USA", 0.22),
    ("UAE", 0.12),
    ("Germany", 0.10),
    ("France", 0.09),
    ("Australia", 0.08),
    ("Canada", 0.07),
    ("Ireland", 0.07),
    ("Netherlands", 0.06),
    ("Saudi Arabia", 0.05),
    ("China", 0.06),
    ("Japan", 0.04),
    ("Singapore", 0.04),
]

# Base annual trade values (£ million) per HS code for imports and exports (circa 2019)
_BASE_IMPORT_VALUES: dict[str, float] = {
    "3303": 550.0,
    "3304": 820.0,
    "3305": 310.0,
    "3306": 180.0,
    "3307": 240.0,
    "3401": 160.0,
}

_BASE_EXPORT_VALUES: dict[str, float] = {
    "3303": 420.0,
    "3304": 580.0,
    "3305": 195.0,
    "3306": 120.0,
    "3307": 175.0,
    "3401": 130.0,
}

# kg per £1,000 of trade (average density proxy)
_KG_PER_THOUSAND_GBP: dict[str, float] = {
    "3303": 0.40,
    "3304": 0.75,
    "3305": 0.90,
    "3306": 1.10,
    "3307": 0.95,
    "3401": 1.50,
}

# Seasonal adjustment per month (index 0 = Jan)
_SEASONAL: list[float] = [
    0.88, 0.85, 0.95, 1.00, 1.02, 1.03,
    0.98, 0.97, 1.02, 1.05, 1.12, 1.18,
]


# ---------------------------------------------------------------------------
# Demo generation
# ---------------------------------------------------------------------------

def generate_demo_trade_data(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic UK cosmetics trade data (monthly 2015-2024).

    Simulates:
    - Monthly import & export values per HS code and partner country.
    - CAGR of ~4 % for imports and ~3 % for exports.
    - Seasonal variation (higher trade in Q4).
    - COVID dip in 2020 (-15 % for 6 months).
    - Brexit disruption in 2021 (minor reduction in EU imports/exports).

    Args:
        seed: NumPy random seed for reproducibility.

    Returns:
        Tidy DataFrame with one row per (period, hs_code, flow, partner) combination.
    """
    rng = np.random.default_rng(seed)
    periods = pd.period_range(start="2015-01", end="2024-12", freq="M")
    records: list[dict] = []

    for hs_code in HS_CODES_COSMETICS:
        description = _HS_DESCRIPTIONS.get(hs_code, f"HS {hs_code}")
        kg_rate = _KG_PER_THOUSAND_GBP.get(hs_code, 1.0)
        base_year = 2019

        for flow, partners, base_values in [
            ("import", _IMPORT_PARTNERS, _BASE_IMPORT_VALUES),
            ("export", _EXPORT_PARTNERS, _BASE_EXPORT_VALUES),
        ]:
            base_annual = base_values.get(hs_code, 200.0) * 1_000_000  # to GBP
            cagr = 0.04 if flow == "import" else 0.03

            partner_countries = [p[0] for p in partners]
            partner_weights = np.array([p[1] for p in partners])
            partner_weights /= partner_weights.sum()

            for i, period in enumerate(periods):
                year_diff = period.year + (period.month - 1) / 12 - base_year
                trend = base_annual * ((1 + cagr) ** year_diff)

                # Seasonal factor
                seasonal = _SEASONAL[period.month - 1]

                # COVID impact: Apr-Sep 2020 = -15%
                covid_factor = 1.0
                if period.year == 2020 and 3 <= period.month <= 9:
                    covid_factor = 0.85

                # Brexit disruption: Jan-Jun 2021 EU flows reduced by 8%
                brexit_factor = 1.0

                total_monthly = trend / 12 * seasonal * covid_factor * brexit_factor
                noise = rng.normal(1.0, 0.04)
                total_monthly *= noise

                # Distribute across partners
                month_alloc = rng.dirichlet(partner_weights * 20)

                for partner, alloc in zip(partner_countries, month_alloc):
                    partner_value = total_monthly * alloc

                    # Apply Brexit factor to EU partners
                    eu_partners = {"France", "Germany", "Italy", "Netherlands", "Poland", "Belgium"}
                    if (
                        partner in eu_partners
                        and period.year == 2021
                        and period.month <= 6
                    ):
                        brexit_partner_factor = 0.92
                        partner_value *= brexit_partner_factor

                    quantity_kg = partner_value / 1000 * kg_rate

                    records.append(
                        {
                            "period": period.strftime("%Y-%m"),
                            "hs_code": hs_code,
                            "hs_description": description,
                            "flow": flow,
                            "trade_partner_country": partner,
                            "trade_value_gbp": round(partner_value, 2),
                            "net_quantity_kg": round(quantity_kg, 1),
                        }
                    )

    df = pd.DataFrame(records)

    # Pivot to wide import/export value columns per row
    imports = df[df["flow"] == "import"].copy()
    exports = df[df["flow"] == "export"].copy()

    imports = imports.rename(columns={"trade_value_gbp": "import_value_gbp", "net_quantity_kg": "import_quantity_kg"})
    exports = exports.rename(columns={"trade_value_gbp": "export_value_gbp", "net_quantity_kg": "export_quantity_kg"})

    imports = imports.drop(columns=["flow"])
    exports = exports.drop(columns=["flow"])

    # Keep long format but add flow column back for clarity
    logger.info(
        "Generated %d synthetic UK trade records (%d periods, %d HS codes)",
        len(df),
        len(periods),
        len(HS_CODES_COSMETICS),
    )
    return df


# ---------------------------------------------------------------------------
# Real data fetch (HMRC OTS)
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def _try_fetch_hmrc() -> pd.DataFrame | None:
    """Attempt to fetch HMRC Overseas Trade Statistics data.

    HMRC's OTS data is primarily available via bulk download.  This makes a
    best-effort attempt against the public API and returns ``None`` on failure.

    Returns:
        DataFrame on success, ``None`` on any error.
    """
    url = "https://api.uktradeinfo.com/OTS"
    params = {
        "top": 1000,
        "filter": "CommodityId ge 330300 and CommodityId le 340190",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        records = data.get("value", [])
        if not records:
            return None
        return pd.DataFrame(records)
    except Exception as exc:
        logger.warning("HMRC OTS API fetch failed: %s", exc)
        return None


def fetch_trade_data() -> pd.DataFrame:
    """Fetch UK cosmetics trade data, falling back to synthetic data.

    Returns:
        Trade DataFrame.
    """
    if DEMO_MODE:
        logger.info("DEMO_MODE: generating synthetic UK trade data")
        return generate_demo_trade_data()

    result = _try_fetch_hmrc()
    if result is None:
        logger.warning("HMRC OTS data unavailable – using demo data")
        return generate_demo_trade_data()
    return result


def parse_trade_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and types for the trade DataFrame.

    Handles both the demo output format and potential HMRC API format.

    Args:
        raw: Raw trade DataFrame (from demo or real API).

    Returns:
        Standardised trade DataFrame.
    """
    if "period" in raw.columns:
        # Already in standard format from demo generator
        return raw.copy()

    # Attempt to remap HMRC column names to standard schema
    rename_map: dict[str, str] = {
        "MonthId": "period",
        "CommodityId": "hs_code",
        "FlowTypeId": "flow",
        "CountryId": "trade_partner_country",
        "Value": "trade_value_gbp",
        "NetMass": "net_quantity_kg",
    }
    df = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})
    logger.debug("Parsed trade data with %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_trade_data(df: pd.DataFrame) -> Path:
    """Save UK trade data to Parquet.

    Args:
        df: Trade DataFrame.

    Returns:
        :class:`~pathlib.Path` of the written file.
    """
    out_dir = get_data_path("raw/trade")
    out_path = out_dir / "uk_trade_data.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Saved %d trade records to %s", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    """Orchestrate UK trade data ingestion.

    Returns:
        Trade DataFrame.
    """
    logger.info("Starting UK Trade Data ingestion (DEMO_MODE=%s)", DEMO_MODE)
    raw = fetch_trade_data()
    df = parse_trade_data(raw)
    save_trade_data(df)
    logger.info(
        "UK Trade ingestion complete: %d rows, %d HS codes, periods %s–%s",
        len(df),
        df["hs_code"].nunique() if "hs_code" in df.columns else 0,
        df["period"].min() if "period" in df.columns else "?",
        df["period"].max() if "period" in df.columns else "?",
    )
    return df


if __name__ == "__main__":
    run()
