"""
Ingestion script for Open Beauty Facts product data.

API: https://world.openbeautyfacts.org/cgi/search.pl
Fetches cosmetics and beauty product metadata including ingredients, brands,
categories, and packaging information.

DEMO_MODE generates ~200 realistic synthetic product records for UK brands.

Output: ``data/raw/open_beauty_facts/products.json`` (NDJSON)

Usage::

    python -m ingestion.open_beauty_facts
"""

import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.config import (
    DEMO_MODE,
    OPEN_BEAUTY_FACTS_SEARCH_URL,
    get_data_path,
    setup_logging,
)

logger: logging.Logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Demo catalogue data
# ---------------------------------------------------------------------------

_BRANDS: list[str] = [
    "L'Oreal Paris",
    "Dove",
    "Simple",
    "No7",
    "Rimmel",
    "Superdrug",
    "The Ordinary",
    "CeraVe",
    "Neutrogena",
    "Olay",
    "Nivea",
    "Garnier",
    "Maybelline",
    "Revlon",
    "e.l.f. Cosmetics",
    "Revolution Beauty",
    "Pixi",
    "NYX Professional Makeup",
]

_CATEGORIES: list[str] = [
    "face cream",
    "shampoo",
    "conditioner",
    "body lotion",
    "mascara",
    "foundation",
    "serum",
    "sunscreen",
    "face wash",
    "toner",
    "eye cream",
    "lip balm",
    "deodorant",
    "body wash",
    "hair mask",
    "exfoliator",
    "primer",
    "blush",
    "concealer",
    "bronzer",
]

_INGREDIENT_POOLS: dict[str, list[str]] = {
    "face cream": [
        "Aqua", "Glycerin", "Niacinamide", "Hyaluronic Acid", "Shea Butter",
        "Ceramide NP", "Tocopherol", "Retinol", "Allantoin", "Panthenol",
    ],
    "shampoo": [
        "Aqua", "Sodium Laureth Sulfate", "Cocamidopropyl Betaine",
        "Glycerin", "Panthenol", "Dimethicone", "Keratin", "Biotin",
        "Citric Acid", "Sodium Chloride",
    ],
    "conditioner": [
        "Aqua", "Cetearyl Alcohol", "Behentrimonium Chloride",
        "Dimethicone", "Glycerin", "Panthenol", "Argan Oil",
        "Hydrolyzed Keratin", "Lactic Acid",
    ],
    "sunscreen": [
        "Aqua", "Zinc Oxide", "Titanium Dioxide", "Octinoxate",
        "Homosalate", "Avobenzone", "Glycerin", "Tocopherol",
        "Aloe Vera Extract", "Dimethicone",
    ],
    "serum": [
        "Aqua", "Niacinamide", "Zinc PCA", "Hyaluronic Acid",
        "Vitamin C", "Retinol", "Peptides", "Ferulic Acid",
        "Kojic Acid", "Resveratrol",
    ],
    "default": [
        "Aqua", "Glycerin", "Butylene Glycol", "Cetearyl Alcohol",
        "Phenoxyethanol", "Carbomer", "Sodium Hydroxide", "Tocopherol",
        "Parfum", "Citric Acid",
    ],
}

_PACKAGING_TYPES: list[str] = [
    "Plastic bottle",
    "Glass bottle",
    "Pump dispenser",
    "Tube",
    "Jar",
    "Aerosol can",
    "Sachet",
    "Tin",
    "Cardboard box",
]

_PRODUCT_NAME_TEMPLATES: dict[str, list[str]] = {
    "face cream": [
        "{brand} Hydrating Day Cream SPF15",
        "{brand} Intensive Repair Moisturiser",
        "{brand} Nourishing Night Cream",
        "{brand} Brightening Vitamin C Cream",
    ],
    "shampoo": [
        "{brand} Nourishing Argan Shampoo",
        "{brand} Volumising Shampoo 300ml",
        "{brand} Repair & Protect Shampoo",
        "{brand} Colour Protect Shampoo",
    ],
    "conditioner": [
        "{brand} Hydrating Conditioner",
        "{brand} Deep Conditioning Treatment",
        "{brand} Lightweight Leave-In Conditioner",
    ],
    "sunscreen": [
        "{brand} SPF50 Daily Protection",
        "{brand} SPF30 Mineral Sunscreen",
        "{brand} Kids SPF50+ Sunscreen",
        "{brand} SPF50 Invisible Fluid",
    ],
    "serum": [
        "{brand} Hyaluronic Acid Serum",
        "{brand} Vitamin C Brightening Serum",
        "{brand} Retinol 0.5% Serum",
        "{brand} Niacinamide 10% + Zinc 1%",
    ],
    "default": [
        "{brand} Essential {category}",
        "{brand} Advanced {category}",
        "{brand} Pro {category}",
        "{brand} Classic {category}",
    ],
}


# ---------------------------------------------------------------------------
# Demo generation
# ---------------------------------------------------------------------------

def _generate_demo_products(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Open Beauty Facts product records.

    Args:
        n: Number of product records to generate.
        seed: NumPy random seed for reproducibility.

    Returns:
        DataFrame with columns matching the Open Beauty Facts schema.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for i in range(n):
        brand = _BRANDS[rng.integers(0, len(_BRANDS))]
        category = _CATEGORIES[rng.integers(0, len(_CATEGORIES))]

        # Product name
        templates = _PRODUCT_NAME_TEMPLATES.get(category, _PRODUCT_NAME_TEMPLATES["default"])
        template = templates[rng.integers(0, len(templates))]
        product_name = template.format(brand=brand, category=category)

        # Ingredients
        ingredient_pool = _INGREDIENT_POOLS.get(category, _INGREDIENT_POOLS["default"])
        n_ingredients = int(rng.integers(5, len(ingredient_pool) + 1))
        selected = list(rng.choice(ingredient_pool, size=min(n_ingredients, len(ingredient_pool)), replace=False))
        ingredients_text = ", ".join(selected)

        # Packaging
        packaging = _PACKAGING_TYPES[rng.integers(0, len(_PACKAGING_TYPES))]

        # Eco/origin score proxies
        nutriscore = rng.choice(["a", "b", "c", "d", "e", None], p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05])
        eco_score = rng.choice(["a", "b", "c", "d", "e", None], p=[0.05, 0.15, 0.30, 0.30, 0.15, 0.05])

        records.append(
            {
                "product_id": str(uuid4()),
                "product_name": product_name,
                "brands": brand,
                "categories": category,
                "categories_tags": f"en:{category.lower().replace(' ', '-')}",
                "ingredients_text": ingredients_text,
                "packaging": packaging,
                "countries_tags": "en:united-kingdom",
                "nutriscore_grade": nutriscore,
                "eco_score_grade": eco_score,
                "states_tags": "en:complete",
                "last_modified_t": int(pd.Timestamp.now().timestamp()),
            }
        )

    df = pd.DataFrame(records)
    logger.info("Generated %d synthetic Open Beauty Facts products", len(df))
    return df


# ---------------------------------------------------------------------------
# Real API fetch
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_products(page: int = 1, page_size: int = 100) -> dict:
    """Fetch a page of products from the Open Beauty Facts search API.

    Args:
        page: Page number (1-indexed).
        page_size: Number of products per page (max 100 recommended).

    Returns:
        Parsed JSON response dict.

    Raises:
        requests.HTTPError: On non-2xx response.
    """
    params = {
        "action": "process",
        "json": 1,
        "page": page,
        "page_size": page_size,
        "tagtype_0": "countries",
        "tag_contains_0": "contains",
        "tag_0": "united-kingdom",
        "fields": (
            "product_name,brands,categories,categories_tags,"
            "ingredients_text,packaging,countries_tags,"
            "nutriscore_grade,eco_score_grade,states_tags,last_modified_t"
        ),
    }
    logger.info("Fetching Open Beauty Facts page %d (size %d)", page, page_size)
    response = requests.get(OPEN_BEAUTY_FACTS_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_products(raw: dict) -> pd.DataFrame:
    """Parse an Open Beauty Facts API response into a tidy DataFrame.

    Args:
        raw: Parsed JSON response from :func:`fetch_products`.

    Returns:
        Tidy products DataFrame.
    """
    products = raw.get("products", [])
    if not products:
        return pd.DataFrame()

    records: list[dict] = []
    for p in products:
        records.append(
            {
                "product_id": p.get("_id") or str(uuid4()),
                "product_name": p.get("product_name", ""),
                "brands": p.get("brands", ""),
                "categories": p.get("categories", ""),
                "categories_tags": p.get("categories_tags", ""),
                "ingredients_text": p.get("ingredients_text", ""),
                "packaging": p.get("packaging", ""),
                "countries_tags": p.get("countries_tags", ""),
                "nutriscore_grade": p.get("nutriscore_grade"),
                "eco_score_grade": p.get("eco_score_grade"),
                "states_tags": p.get("states_tags", ""),
                "last_modified_t": p.get("last_modified_t"),
            }
        )
    df = pd.DataFrame(records)
    logger.debug("Parsed %d product records", len(df))
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_products(df: pd.DataFrame) -> Path:
    """Save Open Beauty Facts products to NDJSON.

    Args:
        df: Products DataFrame.

    Returns:
        :class:`~pathlib.Path` of the written file.
    """
    out_dir = get_data_path("raw/open_beauty_facts")
    out_path = out_dir / "products.json"

    serialisable = df.copy()
    for col in serialisable.select_dtypes(include=["object"]).columns:
        serialisable[col] = serialisable[col].where(pd.notna(serialisable[col]), None)

    with out_path.open("w", encoding="utf-8") as fh:
        for record in serialisable.to_dict(orient="records"):
            fh.write(json.dumps(record, default=str) + "\n")
    logger.info("Saved %d products to %s", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(max_pages: int = 5) -> pd.DataFrame:
    """Orchestrate Open Beauty Facts ingestion.

    In DEMO_MODE ~200 synthetic product records are generated. In real mode
    up to *max_pages* pages are fetched from the public API.

    Args:
        max_pages: Maximum number of API pages to fetch in real mode.

    Returns:
        Products DataFrame.
    """
    logger.info("Starting Open Beauty Facts ingestion (DEMO_MODE=%s)", DEMO_MODE)

    if DEMO_MODE:
        df = _generate_demo_products()
    else:
        frames: list[pd.DataFrame] = []
        for page in range(1, max_pages + 1):
            raw = fetch_products(page=page)
            df_page = parse_products(raw)
            if df_page.empty:
                logger.info("No more products at page %d – stopping", page)
                break
            frames.append(df_page)
            logger.info("Fetched page %d: %d products", page, len(df_page))
            time.sleep(1.0)  # polite rate limiting

        df = pd.concat(frames, ignore_index=True) if frames else _generate_demo_products()

    df = df.drop_duplicates(subset=["product_id"])
    save_products(df)
    logger.info(
        "Open Beauty Facts ingestion complete: %d products, %d brands",
        len(df),
        df["brands"].nunique(),
    )
    return df


if __name__ == "__main__":
    run()
