"""
Configuration module for HealthBeauty360 ingestion pipeline.

Loads settings from environment variables (with .env support via python-dotenv)
and exposes typed constants, path helpers, and a logging factory.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Runtime flags
# ---------------------------------------------------------------------------
DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() == "true"
USE_GCP: bool = os.getenv("USE_GCP", "false").lower() == "true"

# ---------------------------------------------------------------------------
# GCP / BigQuery
# ---------------------------------------------------------------------------
GCS_BUCKET: str = os.getenv("GCS_BUCKET", "healthbeauty360-raw")
BQ_PROJECT: str = os.getenv("BQ_PROJECT", "healthbeauty360")
BQ_DATASET_BRONZE: str = "bronze"
BQ_DATASET_SILVER: str = "silver"
BQ_DATASET_GOLD: str = "gold"

# ---------------------------------------------------------------------------
# Local data paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))

RAW_DIR: Path = DATA_DIR / "raw"
BRONZE_DIR: Path = DATA_DIR / "bronze"
SILVER_DIR: Path = DATA_DIR / "silver"
GOLD_DIR: Path = DATA_DIR / "gold"

RAW_BANK_HOLIDAYS_DIR: Path = RAW_DIR / "bank_holidays"
RAW_WEATHER_DIR: Path = RAW_DIR / "weather"
RAW_ONS_DIR: Path = RAW_DIR / "ons"
RAW_TRENDS_DIR: Path = RAW_DIR / "google_trends"
RAW_UCI_DIR: Path = RAW_DIR / "uci_retail"
RAW_BEAUTY_FACTS_DIR: Path = RAW_DIR / "open_beauty_facts"
RAW_SHOPIFY_DIR: Path = RAW_DIR / "shopify"
RAW_EBAY_DIR: Path = RAW_DIR / "ebay"
RAW_TRADE_DIR: Path = RAW_DIR / "trade"

# ---------------------------------------------------------------------------
# External API endpoints
# ---------------------------------------------------------------------------
BANK_HOLIDAYS_URL: str = "https://www.gov.uk/bank-holidays.json"

# Open-Meteo historical weather archive
OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"

# ONS Retail Sales Index (BETA API)
ONS_API_BASE: str = "https://api.beta.ons.gov.uk/v1"
ONS_RSI_DATASET: str = "retail-sales-index"
ONS_INTERNET_SALES_DATASET: str = "internet-retail-sales-index"

# UCI Machine Learning Repository
UCI_ONLINE_RETAIL_URL: str = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
)

# Open Beauty Facts
OPEN_BEAUTY_FACTS_SEARCH_URL: str = (
    "https://world.openbeautyfacts.org/cgi/search.pl"
)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
UK_HEALTH_BEAUTY_KEYWORDS: list[str] = [
    "skincare",
    "moisturiser",
    "sunscreen",
    "SPF",
    "vitamin supplements",
    "hair dye",
    "perfume",
    "lip balm",
    "foundation",
    "mascara",
    "serum",
    "body lotion",
    "shampoo",
    "conditioner",
    "deodorant",
    "face wash",
    "eye cream",
    "toner",
    "exfoliator",
    "nail polish",
]

# lat/lon for Open-Meteo queries
UK_WEATHER_LOCATIONS: dict[str, dict[str, float]] = {
    "London": {"lat": 51.5074, "lon": -0.1278},
    "Manchester": {"lat": 53.4808, "lon": -2.2426},
    "Birmingham": {"lat": 52.4862, "lon": -1.8904},
}

# HMRC HS codes for cosmetics & toiletries
HS_CODES_COSMETICS: list[str] = [
    "3303",  # perfumes & toilet waters
    "3304",  # beauty / make-up preparations
    "3305",  # hair preparations
    "3306",  # oral / dental hygiene
    "3307",  # shaving / bath preparations, deodorants
    "3401",  # soap & organic surface-active products
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_data_path(subdir: str) -> Path:
    """Return an absolute Path for *subdir* under DATA_DIR, creating it if needed.

    Args:
        subdir: Relative sub-directory name (e.g. ``"raw/weather"``).

    Returns:
        Resolved :class:`~pathlib.Path` that is guaranteed to exist on disk.
    """
    path = DATA_DIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(name: str) -> logging.Logger:
    """Create and return a configured :class:`logging.Logger`.

    Uses a single ``StreamHandler`` with a human-readable format. Safe to call
    multiple times – duplicate handlers are not added.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG if DEMO_MODE else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger
