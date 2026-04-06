"""Synthetic data generation configuration for HealthBeauty360."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Volume constants
# ---------------------------------------------------------------------------
N_SKUS: int = 500
N_CUSTOMERS: int = 10_000
N_TRANSACTIONS: int = 50_000
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# UK Health & Beauty categories
# ---------------------------------------------------------------------------
UK_CATEGORIES: list[str] = [
    "skincare",
    "haircare",
    "fragrance",
    "bath_body",
    "vitamins_supplements",
    "makeup",
    "sun_care",
]

# ---------------------------------------------------------------------------
# Realistic UK / international health & beauty brands
# ---------------------------------------------------------------------------
UK_BRANDS: list[str] = [
    "The Body Shop",
    "Boots",
    "Superdrug Own Brand",
    "L'Oréal",
    "Nivea",
    "Dove",
    "Garnier",
    "Simple",
    "Elemis",
    "REN Clean Skincare",
    "NARS",
    "Charlotte Tilbury",
    "Clarins",
    "Neutrogena",
    "Head & Shoulders",
    "Pantene",
    "Radox",
    "Olaz (Olay)",
    "Liz Earle",
    "Holland & Barrett Own Brand",
]

# ---------------------------------------------------------------------------
# Fictitious suppliers (name + ID)
# ---------------------------------------------------------------------------
UK_SUPPLIERS: list[dict] = [
    {"supplier_id": "SUP001", "supplier_name": "BrightChem Logistics Ltd"},
    {"supplier_id": "SUP002", "supplier_name": "Amber Beauty Wholesale"},
    {"supplier_id": "SUP003", "supplier_name": "Northern Essence Dist."},
    {"supplier_id": "SUP004", "supplier_name": "PurePack UK Ltd"},
    {"supplier_id": "SUP005", "supplier_name": "Greenleaf Health Supplies"},
    {"supplier_id": "SUP006", "supplier_name": "SilkLine Trade Co."},
    {"supplier_id": "SUP007", "supplier_name": "ClearWater FMCG Ltd"},
    {"supplier_id": "SUP008", "supplier_name": "Pinnacle Beauty Group"},
    {"supplier_id": "SUP009", "supplier_name": "Meridian Cosmetics Dist."},
    {"supplier_id": "SUP010", "supplier_name": "Apex Health Imports Ltd"},
]

# ---------------------------------------------------------------------------
# Price ranges (GBP) by category: (min, max)
# ---------------------------------------------------------------------------
PRICE_RANGES: dict[str, tuple[float, float]] = {
    "skincare": (6.99, 89.99),
    "haircare": (4.99, 49.99),
    "fragrance": (19.99, 149.99),
    "bath_body": (3.99, 29.99),
    "vitamins_supplements": (5.99, 39.99),
    "makeup": (7.99, 69.99),
    "sun_care": (5.99, 34.99),
}

# ---------------------------------------------------------------------------
# Lead-time ranges (days) by category: (min_days, max_days)
# ---------------------------------------------------------------------------
LEAD_TIME_RANGES: dict[str, tuple[int, int]] = {
    "skincare": (7, 28),
    "haircare": (5, 21),
    "fragrance": (14, 42),
    "bath_body": (5, 21),
    "vitamins_supplements": (7, 35),
    "makeup": (7, 28),
    "sun_care": (5, 21),
}

# ---------------------------------------------------------------------------
# Marketplace fee structure
# ---------------------------------------------------------------------------
MARKETPLACE_FEES: dict[str, float] = {
    "amazon": 0.1545,         # Amazon referral fee rate
    "ebay": 0.1280,           # eBay final value fee rate
    "shopify_rate": 0.0290,   # Shopify payment processing rate
    "shopify_fixed": 0.30,    # Shopify fixed transaction fee (GBP)
}

# ---------------------------------------------------------------------------
# HS codes by category (UK Trade Tariff headings)
# ---------------------------------------------------------------------------
HS_CODES: dict[str, str] = {
    "skincare": "3304",
    "haircare": "3305",
    "fragrance": "3303",
    "bath_body": "3307",
    "vitamins_supplements": "2106",
    "makeup": "3304",
    "sun_care": "3304",
}

# ---------------------------------------------------------------------------
# Import duty rates by HS heading (UK Global Tariff, %)
# ---------------------------------------------------------------------------
IMPORT_DUTY_RATES: dict[str, float] = {
    "3303": 0.067,   # perfumes / toilet waters
    "3304": 0.067,   # beauty / make-up preparations
    "3305": 0.067,   # hair preparations
    "3307": 0.067,   # shaving / bath preparations
    "2106": 0.050,   # food supplements
    "3401": 0.050,   # soap / organic surface-active products
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path(__file__).parent.parent / "data"


def get_synthetic_path(filename: str) -> Path:
    """Return the absolute path for a synthetic output file.

    Parameters
    ----------
    filename:
        Filename (e.g. ``"inventory.parquet"``).

    Returns
    -------
    Path
        ``data/synthetic/<filename>``
    """
    out_dir = DATA_DIR / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename
