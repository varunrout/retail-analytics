"""
Synthetic cost data generator.

Generates landed cost, marketplace fees, fulfilment costs, and margin
estimates for each SKU.  Writes to ``data/synthetic/costs.parquet``.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic.config import (
    HS_CODES,
    IMPORT_DUTY_RATES,
    MARKETPLACE_FEES,
    N_SKUS,
    RANDOM_SEED,
    get_synthetic_path,
)
from synthetic.generate_inventory import generate_sku_master

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fulfilment & packaging cost ranges (GBP) by category
# ---------------------------------------------------------------------------
_FULFIL_RANGES: dict[str, tuple[float, float]] = {
    "skincare": (3.50, 6.50),
    "haircare": (3.00, 6.00),
    "fragrance": (4.50, 8.50),
    "bath_body": (2.50, 5.50),
    "vitamins_supplements": (2.50, 5.00),
    "makeup": (3.00, 7.00),
    "sun_care": (2.50, 5.50),
}

_PACKAGING_RANGES: dict[str, tuple[float, float]] = {
    "skincare": (0.50, 1.50),
    "haircare": (0.30, 1.00),
    "fragrance": (0.80, 1.50),
    "bath_body": (0.20, 0.80),
    "vitamins_supplements": (0.20, 0.70),
    "makeup": (0.40, 1.20),
    "sun_care": (0.20, 0.70),
}

# Weight (grams) per unit by category — drives freight cost
_WEIGHT_RANGES_G: dict[str, tuple[int, int]] = {
    "skincare": (50, 300),
    "haircare": (150, 600),
    "fragrance": (80, 400),
    "bath_body": (200, 800),
    "vitamins_supplements": (60, 350),
    "makeup": (20, 200),
    "sun_care": (100, 500),
}

# Markdown / clearance cost as % of revenue (0–15 %)
_MARKDOWN_RANGE: tuple[float, float] = (0.00, 0.15)


def generate_costs(sku_df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate cost structure for each SKU.

    Parameters
    ----------
    sku_df:
        SKU master returned by
        :func:`synthetic.generate_inventory.generate_sku_master`.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns:

        ``sku_id``, ``product_name``, ``category``, ``hs_code``,
        ``unit_cost_gbp``, ``unit_price_gbp``,
        ``import_duty_pct``, ``import_duty_gbp``,
        ``weight_grams``, ``freight_cost_gbp``,
        ``landed_cost_gbp``,
        ``amazon_fee_gbp``, ``ebay_fee_gbp``, ``shopify_fee_gbp``,
        ``fulfilment_cost_gbp``, ``packaging_cost_gbp``,
        ``markdown_cost_gbp``,
        ``total_cost_gbp``, ``gross_margin_gbp``, ``gross_margin_pct``.
    """
    rng = np.random.default_rng(seed)
    n = len(sku_df)
    categories = sku_df["category"].values
    unit_costs = sku_df["unit_cost_gbp"].values
    unit_prices = sku_df["unit_price_gbp"].values

    # HS codes and import duty rates
    hs_codes = np.array([HS_CODES[cat] for cat in categories])
    import_duty_pcts = np.array([IMPORT_DUTY_RATES[HS_CODES[cat]] for cat in categories])
    import_duty_gbp = (unit_costs * import_duty_pcts).round(4)

    # Weight-based freight cost (£0.50 – £3.00 per unit; scales with weight)
    weight_grams = np.array(
        [
            rng.integers(_WEIGHT_RANGES_G[categories[i]][0], _WEIGHT_RANGES_G[categories[i]][1] + 1)
            for i in range(n)
        ],
        dtype=float,
    )
    # £0.50 base + £0.004 per gram, capped at £3.00
    freight_cost_gbp = np.clip(0.50 + weight_grams * 0.004, 0.50, 3.00).round(4)

    # Landed cost
    landed_cost_gbp = (unit_costs + import_duty_gbp + freight_cost_gbp).round(4)

    # Marketplace fees (computed on unit_price)
    amazon_fee_gbp = (unit_prices * MARKETPLACE_FEES["amazon"]).round(4)
    ebay_fee_gbp = (unit_prices * MARKETPLACE_FEES["ebay"]).round(4)
    shopify_fee_gbp = (
        unit_prices * MARKETPLACE_FEES["shopify_rate"] + MARKETPLACE_FEES["shopify_fixed"]
    ).round(4)

    # Fulfilment cost — random within category range
    fulfilment_cost_gbp = np.array(
        [
            rng.uniform(_FULFIL_RANGES[categories[i]][0], _FULFIL_RANGES[categories[i]][1])
            for i in range(n)
        ]
    ).round(4)

    # Packaging cost — random within category range
    packaging_cost_gbp = np.array(
        [
            rng.uniform(_PACKAGING_RANGES[categories[i]][0], _PACKAGING_RANGES[categories[i]][1])
            for i in range(n)
        ]
    ).round(4)

    # Markdown / clearance cost (% of revenue)
    markdown_pct = rng.uniform(_MARKDOWN_RANGE[0], _MARKDOWN_RANGE[1], size=n)
    markdown_cost_gbp = (unit_prices * markdown_pct).round(4)

    # Total cost uses amazon fee as the reference channel fee (most conservative)
    total_cost_gbp = (
        landed_cost_gbp
        + amazon_fee_gbp
        + fulfilment_cost_gbp
        + packaging_cost_gbp
        + markdown_cost_gbp
    ).round(4)

    gross_margin_gbp = (unit_prices - total_cost_gbp).round(4)
    # Clamp margin percentage to avoid divide-by-zero artefacts
    gross_margin_pct = np.where(
        unit_prices > 0,
        gross_margin_gbp / unit_prices,
        0.0,
    ).round(4)

    df = pd.DataFrame(
        {
            "sku_id": sku_df["sku_id"].values,
            "product_name": sku_df["product_name"].values,
            "category": categories,
            "brand": sku_df["brand"].values,
            "hs_code": hs_codes,
            "unit_cost_gbp": unit_costs.round(2),
            "unit_price_gbp": unit_prices.round(2),
            "import_duty_pct": import_duty_pcts,
            "import_duty_gbp": import_duty_gbp,
            "weight_grams": weight_grams.astype(int),
            "freight_cost_gbp": freight_cost_gbp,
            "landed_cost_gbp": landed_cost_gbp,
            "amazon_fee_gbp": amazon_fee_gbp,
            "ebay_fee_gbp": ebay_fee_gbp,
            "shopify_fee_gbp": shopify_fee_gbp,
            "fulfilment_cost_gbp": fulfilment_cost_gbp,
            "packaging_cost_gbp": packaging_cost_gbp,
            "markdown_cost_gbp": markdown_cost_gbp,
            "total_cost_gbp": total_cost_gbp,
            "gross_margin_gbp": gross_margin_gbp,
            "gross_margin_pct": gross_margin_pct,
        }
    )
    logger.info(
        "Generated costs: %d rows | avg margin=%.1f%%",
        len(df),
        df["gross_margin_pct"].mean() * 100,
    )
    return df


def save_costs(df: pd.DataFrame) -> Path:
    """Save costs DataFrame to ``data/synthetic/costs.parquet``.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`generate_costs`.

    Returns
    -------
    Path
        Absolute path of the written parquet file.
    """
    out_path = get_synthetic_path("costs.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Saved costs to %s (%d rows).", out_path, len(df))
    return out_path


def run() -> pd.DataFrame:
    """Orchestrate cost generation and persist to disk.

    Returns
    -------
    pd.DataFrame
        The full costs DataFrame.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sku_df = generate_sku_master(seed=RANDOM_SEED)
    cost_df = generate_costs(sku_df, seed=RANDOM_SEED)
    save_costs(cost_df)
    return cost_df


if __name__ == "__main__":
    result = run()
    print(result.head())
