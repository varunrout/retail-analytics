"""
Synthetic inventory data generator.

Generates stock_on_hand, safety_stock, reorder_point, supplier info
for ~500 SKUs and writes the result to ``data/synthetic/inventory.parquet``.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from scipy.stats import norm

from synthetic.config import (
    LEAD_TIME_RANGES,
    N_SKUS,
    PRICE_RANGES,
    RANDOM_SEED,
    UK_BRANDS,
    UK_CATEGORIES,
    UK_SUPPLIERS,
    get_synthetic_path,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demand parameters per ABC class
# ---------------------------------------------------------------------------
_ABC_LAMBDA: dict[str, int] = {"A": 150, "B": 80, "C": 30}
_ABC_DEMAND: dict[str, float] = {"A": 12.0, "B": 5.0, "C": 1.5}
_ABC_WEIGHTS: list[float] = [0.20, 0.30, 0.50]   # A / B / C proportions


def generate_sku_master(n_skus: int = N_SKUS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate base SKU master with product metadata.

    Parameters
    ----------
    n_skus:
        Number of synthetic SKUs to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``sku_id``, ``product_name``, ``category``, ``brand``,
        ``unit_cost_gbp``, ``unit_price_gbp``, ``abc_class``,
        ``avg_daily_demand``.
    """
    rng = np.random.default_rng(seed)
    fake = Faker("en_GB")
    fake.seed_instance(seed)

    sku_ids = [f"SKU{str(i).zfill(5)}" for i in range(1, n_skus + 1)]
    categories = rng.choice(UK_CATEGORIES, size=n_skus, replace=True)
    brands = rng.choice(UK_BRANDS, size=n_skus, replace=True)

    # Assign ABC class according to specified proportions
    abc_classes = rng.choice(["A", "B", "C"], size=n_skus, replace=True, p=_ABC_WEIGHTS)

    # Generate unit prices from category price ranges
    unit_prices = np.array(
        [
            rng.uniform(PRICE_RANGES[cat][0], PRICE_RANGES[cat][1])
            for cat in categories
        ]
    )

    # Unit cost is ~45-65 % of price depending on ABC class
    margin_factor_map = {"A": 0.42, "B": 0.52, "C": 0.60}
    unit_costs = np.array(
        [
            unit_prices[i] * margin_factor_map[abc_classes[i]] * rng.uniform(0.9, 1.1)
            for i in range(n_skus)
        ]
    )

    # Average daily demand driven by ABC class with small jitter
    avg_daily_demand = np.array(
        [
            max(0.1, _ABC_DEMAND[abc_classes[i]] * rng.uniform(0.7, 1.3))
            for i in range(n_skus)
        ]
    )

    # Build human-readable product names
    adjectives = [
        "Radiant", "Pure", "Soft", "Fresh", "Glow", "Silky", "Revive",
        "Calm", "Bright", "Nourish", "Repair", "Boost", "Ultra", "Gentle",
        "Smooth", "Hydra", "Renew", "Soothe", "Vital", "Clear",
    ]
    product_types: dict[str, list[str]] = {
        "skincare": ["Moisturiser", "Serum", "Eye Cream", "Face Wash", "Toner", "Exfoliator"],
        "haircare": ["Shampoo", "Conditioner", "Hair Mask", "Scalp Serum", "Leave-in Spray"],
        "fragrance": ["Eau de Parfum", "Eau de Toilette", "Body Mist", "Perfume Oil"],
        "bath_body": ["Body Lotion", "Shower Gel", "Bath Salts", "Body Scrub", "Hand Cream"],
        "vitamins_supplements": ["Multivitamin", "Omega-3 Capsules", "Collagen Tablets",
                                 "Vitamin D Gummies", "Probiotic Capsules"],
        "makeup": ["Foundation", "Mascara", "Lipstick", "Blush", "Eye Shadow Palette", "Primer"],
        "sun_care": ["SPF 50 Sunscreen", "After-Sun Lotion", "SPF 30 Daily Moisturiser",
                     "Self-Tan Mousse", "Sun Protect Spray"],
    }

    product_names = []
    for i in range(n_skus):
        cat = categories[i]
        adj = adjectives[rng.integers(0, len(adjectives))]
        ptype = product_types[cat][rng.integers(0, len(product_types[cat]))]
        product_names.append(f"{brands[i]} {adj} {ptype}")

    df = pd.DataFrame(
        {
            "sku_id": sku_ids,
            "product_name": product_names,
            "category": categories,
            "brand": brands,
            "unit_cost_gbp": unit_costs.round(2),
            "unit_price_gbp": unit_prices.round(2),
            "abc_class": abc_classes,
            "avg_daily_demand": avg_daily_demand.round(4),
        }
    )
    logger.info("Generated SKU master with %d rows.", len(df))
    return df


def generate_inventory(sku_df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate inventory positions for each SKU.

    Parameters
    ----------
    sku_df:
        SKU master returned by :func:`generate_sku_master`.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per SKU with stock metrics, supplier assignment, and
        derived flags.
    """
    rng = np.random.default_rng(seed)

    n = len(sku_df)
    categories = sku_df["category"].values
    abc_classes = sku_df["abc_class"].values
    avg_daily_demand = sku_df["avg_daily_demand"].values

    # Assign lead times from category-specific uniform distributions
    lead_time_days = np.array(
        [
            rng.integers(
                LEAD_TIME_RANGES[categories[i]][0],
                LEAD_TIME_RANGES[categories[i]][1] + 1,
            )
            for i in range(n)
        ],
        dtype=float,
    )

    # Stock on hand — Poisson with lambda driven by ABC class
    stock_on_hand = np.array(
        [
            rng.poisson(lam=_ABC_LAMBDA[abc_classes[i]])
            for i in range(n)
        ],
        dtype=int,
    )

    # Demand variability — CoV ~30 % of avg_daily_demand
    demand_std = avg_daily_demand * 0.30

    # Safety stock: z=1.645 (95th percentile service level)
    z = norm.ppf(0.95)
    safety_stock = np.ceil(z * demand_std * np.sqrt(lead_time_days)).astype(int)

    # Reorder point = safety_stock + avg_daily_demand * lead_time
    reorder_point = np.ceil(safety_stock + avg_daily_demand * lead_time_days).astype(int)

    # Days of cover
    days_cover = (stock_on_hand / np.maximum(avg_daily_demand, 0.01)).round(1)

    # Assign suppliers randomly
    supplier_indices = rng.integers(0, len(UK_SUPPLIERS), size=n)
    supplier_ids = [UK_SUPPLIERS[i]["supplier_id"] for i in supplier_indices]
    supplier_names = [UK_SUPPLIERS[i]["supplier_name"] for i in supplier_indices]

    # Derived flags
    is_stockout = stock_on_hand == 0
    is_overstock = days_cover > 90
    is_below_reorder = (stock_on_hand <= reorder_point) & (~is_stockout)

    inv_df = pd.DataFrame(
        {
            "sku_id": sku_df["sku_id"].values,
            "product_name": sku_df["product_name"].values,
            "category": categories,
            "brand": sku_df["brand"].values,
            "abc_class": abc_classes,
            "unit_cost_gbp": sku_df["unit_cost_gbp"].values,
            "unit_price_gbp": sku_df["unit_price_gbp"].values,
            "avg_daily_demand": avg_daily_demand,
            "lead_time_days": lead_time_days.astype(int),
            "demand_std": demand_std.round(4),
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "stock_on_hand": stock_on_hand,
            "days_cover": days_cover,
            "supplier_id": supplier_ids,
            "supplier_name": supplier_names,
            "is_stockout": is_stockout,
            "is_overstock": is_overstock,
            "is_below_reorder": is_below_reorder,
        }
    )
    logger.info(
        "Generated inventory: %d rows | stockouts=%d | overstock=%d",
        len(inv_df),
        inv_df["is_stockout"].sum(),
        inv_df["is_overstock"].sum(),
    )
    return inv_df


def save_inventory(df: pd.DataFrame) -> Path:
    """Save inventory DataFrame to ``data/synthetic/inventory.parquet``.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`generate_inventory`.

    Returns
    -------
    Path
        Absolute path of the written parquet file.
    """
    out_path = get_synthetic_path("inventory.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Saved inventory to %s (%d rows).", out_path, len(df))
    return out_path


def run(n_skus: int = N_SKUS) -> pd.DataFrame:
    """Orchestrate SKU master + inventory generation and persist to disk.

    Parameters
    ----------
    n_skus:
        Number of SKUs to generate.

    Returns
    -------
    pd.DataFrame
        The full inventory DataFrame (merged with SKU master).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sku_df = generate_sku_master(n_skus=n_skus, seed=RANDOM_SEED)
    inv_df = generate_inventory(sku_df, seed=RANDOM_SEED)
    save_inventory(inv_df)
    return inv_df


if __name__ == "__main__":
    result = run()
    print(result.head())
