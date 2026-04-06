"""
eBay listings ingestion.

DEMO_MODE generates synthetic eBay-style health & beauty listings.
Real mode uses the eBay Browse API.

Output: ``data/raw/ebay/listings.parquet``

Usage::

    python -m ingestion.ebay_listings
"""

import json
import logging
import os
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
    "Skincare",
    "Hair Care",
    "Bath & Body",
    "Fragrances",
    "Vitamins & Supplements",
    "Oral Care",
    "Sun Care",
    "Make-Up",
    "Nail Care",
    "Men's Grooming",
]

_CONDITIONS: list[str] = ["New", "New", "New", "Used", "For parts or not working"]
_CONDITION_WEIGHTS: list[float] = [0.65, 0.15, 0.10, 0.08, 0.02]

_UK_LOCATIONS: list[str] = [
    "London",
    "Manchester",
    "Birmingham",
    "Leeds",
    "Glasgow",
    "Liverpool",
    "Newcastle upon Tyne",
    "Sheffield",
    "Bristol",
    "Edinburgh",
    "Leicester",
    "Coventry",
    "Bradford",
    "Nottingham",
    "Southampton",
    "Cardiff",
    "Belfast",
    "Brighton",
    "Hull",
    "Derby",
]

_TITLE_TEMPLATES: dict[str, list[str]] = {
    "Skincare": [
        "{brand} Hyaluronic Acid Serum 30ml NEW",
        "{brand} Vitamin C Brightening Cream BNIB",
        "{brand} Retinol Night Cream 50ml",
        "{brand} SPF50 Daily Moisturiser",
        "{brand} Niacinamide Toner 200ml",
        "Job Lot {brand} Skincare Bundle x5",
    ],
    "Hair Care": [
        "{brand} Keratin Shampoo & Conditioner Set",
        "{brand} Argan Oil Hair Mask 250ml",
        "{brand} Dry Shampoo 200ml x3 Bundle",
        "{brand} Bond Repair Treatment",
        "{brand} Purple Toning Shampoo 300ml",
    ],
    "Bath & Body": [
        "{brand} Luxury Bath Bomb Set 12pc NEW",
        "{brand} Shea Body Butter 200ml",
        "{brand} Exfoliating Scrub 300g",
        "{brand} Gift Set Shower Gel & Lotion",
        "{brand} Soap Bar Set 6pc Handmade",
    ],
    "Fragrances": [
        "{brand} Eau de Parfum 50ml UNOPENED",
        "{brand} Perfume Gift Set NEW SEALED",
        "{brand} Aftershave 100ml EDT",
        "{brand} Body Mist 200ml x2",
        "{brand} Solid Perfume Tin",
    ],
    "Vitamins & Supplements": [
        "{brand} Vitamin D3 2000IU 180 Tabs",
        "{brand} Omega-3 Fish Oil 1000mg 90 Caps",
        "{brand} Collagen Beauty Powder 250g",
        "{brand} Probiotics 50 Billion 60 Caps",
        "{brand} Multivitamin Women 90 Tabs",
        "Bulk Buy {brand} Vitamins Job Lot x10",
    ],
    "Oral Care": [
        "{brand} Whitening Toothpaste 100ml x4",
        "{brand} Electric Toothbrush Heads 8pc",
        "{brand} Teeth Whitening Kit",
        "{brand} Water Flosser Tips 12pc",
        "{brand} Charcoal Toothpaste",
    ],
    "Sun Care": [
        "{brand} SPF50 Sunscreen 200ml NEW",
        "{brand} Self-Tan Mousse Medium 200ml",
        "{brand} After Sun Aloe Vera Gel 400ml",
        "{brand} Kids SPF50+ 150ml 2-pack",
        "{brand} Sunscreen SPF30 Bundle x3",
    ],
    "Make-Up": [
        "{brand} Eyeshadow Palette 24 Shades",
        "{brand} Foundation Bundle Mixed Shades",
        "{brand} Lip Gloss Set 8pc",
        "{brand} Mascara Volumising NEW",
        "{brand} Make-Up Collection Job Lot",
    ],
    "Nail Care": [
        "{brand} Gel Nail Polish Set 12 Colours",
        "{brand} Nail Strengthener Treatment",
        "{brand} Nail Polish Remover 100ml x3",
    ],
    "Men's Grooming": [
        "{brand} Beard Oil & Balm Gift Set",
        "{brand} Shaving Foam 200ml x4",
        "{brand} Moisturiser for Men SPF15",
        "{brand} Aftershave Balm 100ml",
    ],
    "default": [
        "{brand} Health & Beauty Product Bundle",
        "{brand} Assorted Beauty Items Lot",
    ],
}

_BRANDS: list[str] = [
    "L'Oreal", "Dove", "No7", "Nivea", "The Ordinary", "CeraVe",
    "Neutrogena", "Olay", "Garnier", "Rimmel", "Maybelline",
    "Revolution", "Pixi", "Elemis", "Medik8", "Simple", "Bulldog",
    "Holland & Barrett", "Solgar", "Nature's Best",
]

# (price_min, price_max) in GBP per category
_PRICE_RANGES: dict[str, tuple[float, float]] = {
    "Skincare":              (3.99,  75.00),
    "Hair Care":             (2.49,  35.00),
    "Bath & Body":           (1.99,  30.00),
    "Fragrances":            (5.99,  90.00),
    "Vitamins & Supplements": (4.99,  45.00),
    "Oral Care":             (1.99,  25.00),
    "Sun Care":              (2.99,  28.00),
    "Make-Up":               (1.99,  55.00),
    "Nail Care":             (1.49,  20.00),
    "Men's Grooming":        (3.99,  40.00),
}

_FEEDBACK_SCORES: list[int] = [
    10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000
]


# ---------------------------------------------------------------------------
# Demo generation
# ---------------------------------------------------------------------------

def generate_demo_listings(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic eBay-style health & beauty product listings.

    Args:
        n: Number of listing records to generate.
        seed: NumPy random seed for reproducibility.

    Returns:
        DataFrame with columns: ``listing_id``, ``title``, ``price_gbp``,
        ``seller_feedback_score``, ``condition``, ``category``,
        ``item_location``, ``listing_date``, ``end_date``, ``bids``,
        ``buy_it_now``, ``shipping_cost_gbp``.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    listing_id_base = 3_850_000_000

    # Distribute listings across categories
    n_per_cat = n // len(_CATEGORIES)
    remainder = n % len(_CATEGORIES)

    listing_counter = 0
    for cat_idx, category in enumerate(_CATEGORIES):
        count = n_per_cat + (1 if cat_idx < remainder else 0)
        price_min, price_max = _PRICE_RANGES.get(category, (1.99, 50.0))
        templates = _TITLE_TEMPLATES.get(category, _TITLE_TEMPLATES["default"])

        for _ in range(count):
            listing_id = listing_id_base + listing_counter
            listing_counter += 1

            brand = _BRANDS[rng.integers(0, len(_BRANDS))]
            template = templates[rng.integers(0, len(templates))]
            title = template.format(brand=brand)

            price = round(rng.uniform(price_min, price_max), 2)

            condition_weights = np.array(_CONDITION_WEIGHTS)
            condition_idx = rng.choice(len(_CONDITIONS), p=condition_weights)
            condition = _CONDITIONS[condition_idx]

            feedback_score = int(rng.choice(_FEEDBACK_SCORES))

            location = _UK_LOCATIONS[rng.integers(0, len(_UK_LOCATIONS))]

            days_ago = int(rng.integers(0, 90))
            listing_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_date = (datetime.now() + timedelta(days=int(rng.integers(1, 30)))).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Auction vs Buy-It-Now
            buy_it_now = bool(rng.random() > 0.25)
            bids = 0 if buy_it_now else int(rng.integers(0, 25))

            # Shipping
            if rng.random() < 0.35:
                shipping_cost = 0.0  # free shipping
            else:
                shipping_cost = round(rng.choice([1.99, 2.50, 2.99, 3.50, 3.99, 4.99]), 2)

            records.append(
                {
                    "listing_id": listing_id,
                    "title": title,
                    "price_gbp": price,
                    "seller_feedback_score": feedback_score,
                    "condition": condition,
                    "category": category,
                    "item_location": location,
                    "listing_date": listing_date,
                    "end_date": end_date,
                    "bids": bids,
                    "buy_it_now": buy_it_now,
                    "shipping_cost_gbp": shipping_cost,
                }
            )

    df = pd.DataFrame(records)
    logger.info("Generated %d synthetic eBay listings across %d categories", len(df), df["category"].nunique())
    return df


# ---------------------------------------------------------------------------
# Real eBay Browse API fetch
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_ebay_listings(query: str, access_token: str, limit: int = 200) -> list[dict]:
    """Search eBay listings via the Browse API.

    Args:
        query: Free-text search query, e.g. ``"health beauty skincare"``.
        access_token: OAuth 2.0 Bearer token for the eBay API.
        limit: Maximum number of results to retrieve (max 200 per call).

    Returns:
        List of raw item dicts from the eBay API response.

    Raises:
        requests.HTTPError: On non-2xx response.
    """
    url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-EBAY-C-MARKETPLACE-ID": "EBAY_GB",
    }
    params = {
        "q": query,
        "limit": min(limit, 200),
        "filter": "categoryIds:{26395}",  # Health & Beauty category
    }
    logger.info("Fetching eBay listings for query: '%s'", query)
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("itemSummaries", [])
    logger.info("Received %d listings from eBay API", len(items))
    return items


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_listings(raw: list[dict]) -> pd.DataFrame:
    """Parse eBay Browse API item summaries into a tidy DataFrame.

    Args:
        raw: List of eBay item summary dicts.

    Returns:
        Tidy listings DataFrame.
    """
    records: list[dict] = []
    for item in raw:
        price_info = item.get("price", {})
        shipping = item.get("shippingOptions", [{}])[0]
        shipping_cost = float(shipping.get("shippingCost", {}).get("value", 0))

        records.append(
            {
                "listing_id": item.get("itemId"),
                "title": item.get("title"),
                "price_gbp": float(price_info.get("value", 0)),
                "seller_feedback_score": item.get("seller", {}).get("feedbackScore", 0),
                "condition": item.get("condition"),
                "category": item.get("categories", [{}])[0].get("categoryName"),
                "item_location": item.get("itemLocation", {}).get("city"),
                "listing_date": item.get("itemCreationDate"),
                "end_date": item.get("itemEndDate"),
                "bids": item.get("bidCount", 0),
                "buy_it_now": item.get("buyingOptions", []) == ["FIXED_PRICE"],
                "shipping_cost_gbp": shipping_cost,
            }
        )

    df = pd.DataFrame(records)
    logger.debug("Parsed %d eBay listing records", len(df))
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_listings(df: pd.DataFrame) -> Path:
    """Save eBay listings to Parquet.

    Args:
        df: Listings DataFrame.

    Returns:
        :class:`~pathlib.Path` of the written file.
    """
    out_dir = get_data_path("raw/ebay")
    out_path = out_dir / "listings.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Saved %d listings to %s", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    """Orchestrate eBay listings ingestion.

    In DEMO_MODE synthetic listings are generated. In real mode
    ``EBAY_ACCESS_TOKEN`` must be set as an environment variable.

    Returns:
        Listings DataFrame.
    """
    logger.info("Starting eBay listings ingestion (DEMO_MODE=%s)", DEMO_MODE)

    if DEMO_MODE:
        df = generate_demo_listings()
    else:
        access_token = os.getenv("EBAY_ACCESS_TOKEN", "")
        if not access_token:
            logger.warning("EBAY_ACCESS_TOKEN not set – using demo data")
            df = generate_demo_listings()
        else:
            query = "health beauty skincare haircare vitamins"
            raw = fetch_ebay_listings(query, access_token)
            df = parse_listings(raw)
            if df.empty:
                logger.warning("No eBay listings returned – using demo data")
                df = generate_demo_listings()

    save_listings(df)
    logger.info(
        "eBay ingestion complete: %d listings across %d categories",
        len(df),
        df["category"].nunique(),
    )
    return df


if __name__ == "__main__":
    run()
