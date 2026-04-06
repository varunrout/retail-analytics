"""
Shopify product catalogue ingestion.

DEMO_MODE generates 500 synthetic health & beauty SKUs representing a
realistic UK-based Shopify store. Real mode calls the Shopify Admin REST API.

Output: ``data/raw/shopify/products.parquet``

Usage::

    python -m ingestion.shopify_products
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.config import (
    DEMO_MODE,
    get_data_path,
    setup_logging,
)

logger: logging.Logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Demo catalogue parameters
# ---------------------------------------------------------------------------

_CATEGORIES: list[str] = [
    "skincare",
    "haircare",
    "bath_body",
    "fragrance",
    "vitamins",
    "oral_care",
    "sun_care",
]

_VENDORS: list[str] = [
    "No7",
    "The Ordinary",
    "CeraVe",
    "L'Oreal Paris",
    "Dove",
    "Simple",
    "Nivea",
    "Olay",
    "Garnier",
    "Neutrogena",
    "Pixi Beauty",
    "Revolution Beauty",
    "REN Clean Skincare",
    "Elemis",
    "Medik8",
    "Paula's Choice",
    "Weleda",
    "Bulldog Skincare",
    "Dr. Organic",
    "Holland & Barrett",
]

# Price range (min, max) in GBP per category
_PRICE_RANGES: dict[str, tuple[float, float]] = {
    "skincare":   (8.99,  89.99),
    "haircare":   (4.99,  34.99),
    "bath_body":  (3.99,  24.99),
    "fragrance":  (14.99, 79.99),
    "vitamins":   (6.99,  39.99),
    "oral_care":  (2.99,  24.99),
    "sun_care":   (5.99,  29.99),
}

_PRODUCT_TEMPLATES: dict[str, list[str]] = {
    "skincare": [
        "Hydrating Day Cream SPF15",
        "Hyaluronic Acid Serum 30ml",
        "Vitamin C Brightening Serum",
        "Retinol Night Cream",
        "Niacinamide 10% Toner",
        "Micellar Cleansing Water 400ml",
        "Eye Cream Peptide Complex",
        "AHA BHA Exfoliating Peel",
        "Overnight Recovery Mask",
        "Ultra Light Moisturiser SPF30",
    ],
    "haircare": [
        "Repair Shampoo 300ml",
        "Moisturising Conditioner 300ml",
        "Deep Conditioning Hair Mask",
        "Argan Oil Hair Serum 50ml",
        "Dry Shampoo Original 200ml",
        "Purple Toning Shampoo",
        "Bond Repair Treatment",
        "Scalp Exfoliating Scrub",
        "Heat Protect Spray 200ml",
        "Leave-In Cream 150ml",
    ],
    "bath_body": [
        "Lavender Body Butter 200ml",
        "Exfoliating Body Scrub 300g",
        "Bath Bomb Set 6pc",
        "Shower Gel Rose & Neroli 500ml",
        "Shea Body Lotion 400ml",
        "Natural Bar Soap 100g",
        "Epsom Salt Soak 500g",
        "Silky Soft Body Oil 100ml",
    ],
    "fragrance": [
        "Floral Eau de Parfum 50ml",
        "Fresh Citrus Eau de Toilette 75ml",
        "Woody Oriental EDP 30ml",
        "Body Mist Vanilla 200ml",
        "Roll-On Perfume Oil 10ml",
        "Solid Perfume Compact 5g",
    ],
    "vitamins": [
        "Vitamin D3 2000IU 90 Tablets",
        "Omega-3 Fish Oil 1000mg 60 Caps",
        "Collagen Beauty Powder 250g",
        "Probiotics 10 Billion 60 Caps",
        "Magnesium Glycinate 120 Caps",
        "Biotin 10000mcg 60 Tablets",
        "Iron + Vitamin C 60 Tablets",
        "Evening Primrose Oil 1000mg",
    ],
    "oral_care": [
        "Whitening Toothpaste 100ml",
        "Charcoal Mouthwash 500ml",
        "Teeth Whitening Strips 14pc",
        "Electric Toothbrush Heads 4pc",
        "Sensitive Toothpaste Twin Pack",
        "Water Flosser Tips 8pc",
    ],
    "sun_care": [
        "SPF50 Daily Sunscreen 200ml",
        "SPF30 Tinted Face Fluid 40ml",
        "Kids SPF50+ Sunscreen 150ml",
        "After Sun Cooling Gel 200ml",
        "Self-Tan Mousse Medium 200ml",
        "SPF50+ Invisible Spray 150ml",
        "Tan Remover Exfoliating Mitt",
    ],
}

# Inventory level distribution (mean, std) by category
_INVENTORY_DIST: dict[str, tuple[int, int]] = {
    "skincare":   (120, 60),
    "haircare":   (100, 50),
    "bath_body":  (150, 70),
    "fragrance":  (60,  30),
    "vitamins":   (200, 80),
    "oral_care":  (180, 70),
    "sun_care":   (80,  40),
}


# ---------------------------------------------------------------------------
# Demo generation
# ---------------------------------------------------------------------------

def generate_demo_products(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Shopify-style product catalogue.

    Each record corresponds to a single SKU (product variant). Products with
    multiple variants (e.g., different sizes) share the same ``product_id``.

    Args:
        n: Target number of SKU records.
        seed: NumPy random seed for reproducibility.

    Returns:
        DataFrame with Shopify-compatible columns:
        ``product_id``, ``title``, ``vendor``, ``product_type``, ``tags``,
        ``created_at``, ``sku``, ``price_gbp``, ``compare_at_price_gbp``,
        ``inventory_quantity``, ``weight_g``, ``barcode``.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    # Distribute products across categories
    n_per_cat = n // len(_CATEGORIES)
    remainder = n % len(_CATEGORIES)

    product_id_counter = 80_000_000
    sku_counter = 10_000

    for cat_idx, category in enumerate(_CATEGORIES):
        templates = _PRODUCT_TEMPLATES.get(category, ["Product"])
        count = n_per_cat + (1 if cat_idx < remainder else 0)
        price_min, price_max = _PRICE_RANGES[category]
        inv_mean, inv_std = _INVENTORY_DIST[category]

        generated = 0
        while generated < count:
            product_id = product_id_counter
            product_id_counter += 1

            # Number of variants per product (1-3)
            n_variants = int(rng.choice([1, 1, 2, 3], p=[0.5, 0.25, 0.15, 0.10]))

            vendor = _VENDORS[rng.integers(0, len(_VENDORS))]
            template = templates[rng.integers(0, len(templates))]
            title = f"{vendor} {template}"

            # Tags
            tag_list = [category, vendor.lower().replace(" ", "-"), "health-beauty", "uk"]
            if rng.random() < 0.3:
                tag_list.append("new-arrival")
            if rng.random() < 0.2:
                tag_list.append("bestseller")
            tags = ", ".join(tag_list)

            # Created date (last 3 years)
            days_ago = int(rng.integers(1, 1095))
            created_at = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")

            base_price = round(rng.uniform(price_min, price_max), 2)

            for v in range(n_variants):
                if generated >= count:
                    break

                sku = f"{category[:3].upper()}-{sku_counter:05d}"
                sku_counter += 1

                # Variant price: slight variation from base
                price = round(base_price * rng.uniform(0.9, 1.1), 2)
                price = round(round(price / 0.01) * 0.01, 2)  # round to pence

                # Compare-at price (RRP) – ~40 % of products have a sale price
                if rng.random() < 0.4:
                    compare_at = round(price * rng.uniform(1.15, 1.50), 2)
                else:
                    compare_at = None

                inventory = max(0, int(rng.normal(inv_mean, inv_std)))
                weight_g = int(rng.integers(30, 600))
                barcode = f"50{''.join([str(rng.integers(0, 10)) for _ in range(11)])}"

                records.append(
                    {
                        "product_id": product_id,
                        "title": title,
                        "vendor": vendor,
                        "product_type": category,
                        "tags": tags,
                        "created_at": created_at,
                        "sku": sku,
                        "price_gbp": price,
                        "compare_at_price_gbp": compare_at,
                        "inventory_quantity": inventory,
                        "weight_g": weight_g,
                        "barcode": barcode,
                    }
                )
                generated += 1

    df = pd.DataFrame(records[:n])
    logger.info("Generated %d synthetic Shopify SKUs across %d categories", len(df), df["product_type"].nunique())
    return df


# ---------------------------------------------------------------------------
# Real Shopify API fetch
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_shopify_products(store_url: str, access_token: str) -> list[dict]:
    """Fetch all products from a Shopify store via the Admin REST API.

    Args:
        store_url: Store domain, e.g. ``"mystore.myshopify.com"``.
        access_token: Shopify Admin API access token.

    Returns:
        List of raw product dicts from the Shopify API.

    Raises:
        requests.HTTPError: On non-2xx response.
    """
    all_products: list[dict] = []
    url = f"https://{store_url}/admin/api/2023-10/products.json"
    headers = {"X-Shopify-Access-Token": access_token}
    params: dict = {"limit": 250}

    while url:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        all_products.extend(data.get("products", []))
        logger.info("Fetched %d products (total so far: %d)", len(data.get("products", [])), len(all_products))

        # Handle pagination via Link header
        link_header = resp.headers.get("Link", "")
        next_url = None
        if 'rel="next"' in link_header:
            for part in link_header.split(","):
                if 'rel="next"' in part:
                    next_url = part.strip().split(";")[0].strip().strip("<>")
                    break
        url = next_url
        params = {}

    return all_products


def parse_products(raw: list[dict]) -> pd.DataFrame:
    """Flatten Shopify product+variant JSON into a tidy SKU-level DataFrame.

    Args:
        raw: List of Shopify product dicts (each containing a ``variants`` list).

    Returns:
        Flat DataFrame at SKU (variant) level.
    """
    records: list[dict] = []
    for product in raw:
        for variant in product.get("variants", []):
            records.append(
                {
                    "product_id": product.get("id"),
                    "title": product.get("title"),
                    "vendor": product.get("vendor"),
                    "product_type": product.get("product_type"),
                    "tags": product.get("tags"),
                    "created_at": product.get("created_at"),
                    "sku": variant.get("sku"),
                    "price_gbp": float(variant.get("price", 0)),
                    "compare_at_price_gbp": (
                        float(variant["compare_at_price"]) if variant.get("compare_at_price") else None
                    ),
                    "inventory_quantity": variant.get("inventory_quantity"),
                    "weight_g": variant.get("grams"),
                    "barcode": variant.get("barcode"),
                }
            )
    df = pd.DataFrame(records)
    logger.debug("Parsed %d SKU records from %d products", len(df), len(raw))
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_products(df: pd.DataFrame) -> Path:
    """Save Shopify products to Parquet.

    Args:
        df: Products DataFrame.

    Returns:
        :class:`~pathlib.Path` of the written file.
    """
    out_dir = get_data_path("raw/shopify")
    out_path = out_dir / "products.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Saved %d SKUs to %s", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    """Orchestrate Shopify product catalogue ingestion.

    In DEMO_MODE synthetic SKUs are generated. In real mode the store URL
    and access token must be set via environment variables
    ``SHOPIFY_STORE_URL`` and ``SHOPIFY_ACCESS_TOKEN``.

    Returns:
        Products DataFrame.
    """
    import os

    logger.info("Starting Shopify products ingestion (DEMO_MODE=%s)", DEMO_MODE)

    if DEMO_MODE:
        df = generate_demo_products()
    else:
        store_url = os.getenv("SHOPIFY_STORE_URL", "")
        access_token = os.getenv("SHOPIFY_ACCESS_TOKEN", "")
        if not store_url or not access_token:
            logger.warning("SHOPIFY_STORE_URL or SHOPIFY_ACCESS_TOKEN not set – using demo data")
            df = generate_demo_products()
        else:
            raw = fetch_shopify_products(store_url, access_token)
            df = parse_products(raw)

    save_products(df)
    logger.info(
        "Shopify ingestion complete: %d SKUs, %d vendors, %d categories",
        len(df),
        df["vendor"].nunique(),
        df["product_type"].nunique(),
    )
    return df


if __name__ == "__main__":
    run()
