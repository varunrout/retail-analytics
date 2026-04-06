"""
Synthetic enriched transaction generator.

Generates 50,000 transactions in UCI Online Retail II style, then enriches
them with channel attribution, promotional flags, margin estimates, and
seasonal adjustments.

Writes to ``data/synthetic/transactions.parquet``.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic.config import (
    N_TRANSACTIONS,
    RANDOM_SEED,
    UK_CATEGORIES,
    get_synthetic_path,
)
from synthetic.generate_inventory import generate_sku_master

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel mix
# ---------------------------------------------------------------------------
_CHANNELS: list[str] = ["web", "amazon", "ebay", "physical"]
_CHANNEL_WEIGHTS: list[float] = [0.50, 0.25, 0.15, 0.10]

# Gross margin benchmarks by category
_CATEGORY_MARGINS: dict[str, float] = {
    "skincare": 0.55,
    "haircare": 0.48,
    "fragrance": 0.65,
    "bath_body": 0.45,
    "vitamins_supplements": 0.45,
    "makeup": 0.58,
    "sun_care": 0.50,
}

# ---------------------------------------------------------------------------
# Seasonal multipliers per month
# ---------------------------------------------------------------------------
_MONTHLY_VOLUME_FACTOR: dict[int, float] = {
    1: 0.80,   # January dip
    2: 1.00,
    3: 1.05,   # Mother's Day
    4: 0.98,
    5: 1.00,
    6: 1.15,   # Summer starts
    7: 1.15,
    8: 1.15,   # Summer peak
    9: 1.00,
    10: 1.00,
    11: 1.10,  # Pre-Christmas
    12: 1.40,  # Christmas peak
}

# Category-specific seasonal boosts: (month, category, multiplier)
_CATEGORY_SEASONAL: list[tuple[int, str, float]] = [
    (2, "fragrance", 1.10),   # Valentine's
    (3, "skincare", 1.15),    # Mother's Day
    (6, "sun_care", 1.30),
    (7, "sun_care", 1.30),
    (8, "sun_care", 1.25),
    (12, "fragrance", 1.20),  # Christmas gifts
    (12, "skincare", 1.15),
    (12, "makeup", 1.10),
]


def _season_from_month(month: int) -> str:
    """Map a calendar month to a meteorological season."""
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    if month in (9, 10, 11):
        return "Autumn"
    return "Winter"


def generate_base_transactions(
    n: int = N_TRANSACTIONS, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate and enrich synthetic transaction records.

    Generates base transactions in UCI Online Retail II format, then
    enriches each row with channel attribution, promotional flags,
    margin estimates, and seasonal metadata.

    Seasonal patterns applied:

    * **December** (+40 % volume) — Christmas peak
    * **January** (−20 %) — post-Christmas dip
    * **June–August** (+15 % overall; +30 % for sun_care)
    * **February** (+10 % fragrance) — Valentine's Day
    * **March** (+15 % skincare) — Mother's Day

    Parameters
    ----------
    n:
        Number of transactions to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per line item with columns:

        ``invoice_no``, ``stock_code``, ``description``,
        ``quantity``, ``invoice_date``, ``unit_price_gbp``,
        ``customer_id``, ``country``,
        ``category``, ``brand``, ``channel``,
        ``is_promotional``, ``discount_pct``,
        ``gross_sales_gbp``, ``discount_amount_gbp``, ``net_revenue_gbp``,
        ``gross_margin_pct``, ``gross_margin_gbp``,
        ``is_return``, ``season``.
    """
    rng = np.random.default_rng(seed)

    # Load SKU master for realistic product metadata
    sku_df = generate_sku_master(seed=seed)
    n_skus = len(sku_df)

    # -----------------------------------------------------------------------
    # Date generation with seasonal volume weighting
    # -----------------------------------------------------------------------
    # Build a date pool: 2021-01-01 – 2024-08-31
    date_start = pd.Timestamp("2021-01-01")
    date_end = pd.Timestamp("2024-08-31")
    all_days = pd.date_range(date_start, date_end, freq="D")

    # Assign weight to each day by month seasonal factor
    day_weights = np.array(
        [_MONTHLY_VOLUME_FACTOR[d.month] for d in all_days], dtype=float
    )
    day_weights /= day_weights.sum()

    # Draw transaction dates
    chosen_day_indices = rng.choice(len(all_days), size=n, replace=True, p=day_weights)
    invoice_dates = all_days[chosen_day_indices]

    # -----------------------------------------------------------------------
    # SKU assignment — category-seasonal boost implemented via oversampling
    # -----------------------------------------------------------------------
    # Build base SKU weights (uniform by default)
    sku_base_weights = np.ones(n_skus, dtype=float)

    # We'll assign SKUs transaction-by-transaction to account for
    # per-date category boosts, but for performance we do a vectorised
    # approximate approach: compute per-transaction SKU weights based on
    # month of the transaction date.
    months = pd.Series(invoice_dates).dt.month.values  # shape (n,)

    # Map category → SKU indices
    cat_to_idx: dict[str, np.ndarray] = {
        cat: np.where(sku_df["category"].values == cat)[0]
        for cat in UK_CATEGORIES
    }

    # Build month-level category multipliers
    month_cat_boost: dict[int, dict[str, float]] = {m: {} for m in range(1, 13)}
    for month, cat, mult in _CATEGORY_SEASONAL:
        month_cat_boost[month][cat] = mult

    # Assign SKU indices one transaction at a time (vectorised within month)
    sku_indices = np.empty(n, dtype=int)
    for month in range(1, 13):
        mask = months == month
        count = mask.sum()
        if count == 0:
            continue
        weights = sku_base_weights.copy()
        for cat, mult in month_cat_boost[month].items():
            if cat in cat_to_idx and len(cat_to_idx[cat]) > 0:
                weights[cat_to_idx[cat]] *= mult
        weights /= weights.sum()
        sku_indices[mask] = rng.choice(n_skus, size=count, replace=True, p=weights)

    selected_skus = sku_df.iloc[sku_indices].reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Core transaction fields
    # -----------------------------------------------------------------------
    # Invoice numbers (some invoices contain multiple lines)
    n_invoices = max(1, n // 3)
    invoice_pool = [f"INV{str(i).zfill(6)}" for i in range(1, n_invoices + 1)]
    invoice_nos = rng.choice(invoice_pool, size=n, replace=True)

    # Quantity: mostly 1–6, small % bulk orders
    quantity = rng.choice(
        [1, 2, 3, 4, 5, 6, 12, 24],
        size=n,
        replace=True,
        p=[0.40, 0.25, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02],
    )

    # Returns: ~3 % of transactions are negative quantity
    is_return = rng.random(n) < 0.03
    quantity = np.where(is_return, -np.abs(quantity), quantity)

    # Customer IDs (FK to CRM; ~15 % guest checkouts → NaN)
    n_cust = 10_000
    cust_ids_pool = [f"CUST{str(i).zfill(5)}" for i in range(1, n_cust + 1)]
    raw_cust = rng.choice(cust_ids_pool, size=n, replace=True)
    is_guest = rng.random(n) < 0.15
    customer_ids = np.where(is_guest, None, raw_cust)

    # Country: ~85 % United Kingdom
    countries = rng.choice(
        ["United Kingdom", "Republic of Ireland", "Germany", "France", "USA"],
        size=n,
        replace=True,
        p=[0.85, 0.05, 0.04, 0.03, 0.03],
    )

    # -----------------------------------------------------------------------
    # Channel
    # -----------------------------------------------------------------------
    channels = rng.choice(_CHANNELS, size=n, replace=True, p=_CHANNEL_WEIGHTS)

    # -----------------------------------------------------------------------
    # Promotional flags & discounts
    # -----------------------------------------------------------------------
    is_promotional = rng.random(n) < 0.20
    discount_pct = np.where(
        is_promotional,
        rng.uniform(0.05, 0.40, size=n),
        0.0,
    ).round(4)

    # -----------------------------------------------------------------------
    # Revenue calculations
    # -----------------------------------------------------------------------
    unit_price = selected_skus["unit_price_gbp"].values.astype(float)
    abs_quantity = np.abs(quantity).astype(float)
    sign = np.where(is_return, -1, 1)

    gross_sales_gbp = (sign * abs_quantity * unit_price).round(2)
    discount_amount_gbp = (gross_sales_gbp * discount_pct).round(2)
    net_revenue_gbp = (gross_sales_gbp - discount_amount_gbp).round(2)

    # Gross margin: base category margin ± small noise
    base_margins = np.array(
        [_CATEGORY_MARGINS[cat] for cat in selected_skus["category"].values]
    )
    noise = rng.uniform(-0.05, 0.05, size=n)
    gross_margin_pct = np.clip(base_margins + noise, 0.10, 0.80).round(4)
    gross_margin_gbp = (net_revenue_gbp * gross_margin_pct).round(2)

    # -----------------------------------------------------------------------
    # Season label
    # -----------------------------------------------------------------------
    seasons = np.array([_season_from_month(d.month) for d in invoice_dates])

    # -----------------------------------------------------------------------
    # Assemble DataFrame
    # -----------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "invoice_no": invoice_nos,
            "stock_code": selected_skus["sku_id"].values,
            "description": selected_skus["product_name"].values,
            "quantity": quantity,
            "invoice_date": invoice_dates,
            "unit_price_gbp": unit_price.round(2),
            "customer_id": customer_ids,
            "country": countries,
            "category": selected_skus["category"].values,
            "brand": selected_skus["brand"].values,
            "channel": channels,
            "is_promotional": is_promotional,
            "discount_pct": discount_pct,
            "gross_sales_gbp": gross_sales_gbp,
            "discount_amount_gbp": discount_amount_gbp,
            "net_revenue_gbp": net_revenue_gbp,
            "gross_margin_pct": gross_margin_pct,
            "gross_margin_gbp": gross_margin_gbp,
            "is_return": is_return,
            "season": seasons,
        }
    )
    logger.info(
        "Generated %d transactions | returns=%d | promotional=%d | "
        "total net revenue=£{:,.0f}".format(df["net_revenue_gbp"].sum()),
        len(df),
        df["is_return"].sum(),
        df["is_promotional"].sum(),
    )
    return df


def save_transactions(df: pd.DataFrame) -> Path:
    """Save transactions DataFrame to ``data/synthetic/transactions.parquet``.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`generate_base_transactions`.

    Returns
    -------
    Path
        Absolute path of the written parquet file.
    """
    out_path = get_synthetic_path("transactions.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Saved transactions to %s (%d rows).", out_path, len(df))
    return out_path


def run() -> pd.DataFrame:
    """Orchestrate transaction generation and persist to disk.

    Returns
    -------
    pd.DataFrame
        The full transactions DataFrame.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tx_df = generate_base_transactions(seed=RANDOM_SEED)
    save_transactions(tx_df)
    return tx_df


if __name__ == "__main__":
    result = run()
    print(result.head())
