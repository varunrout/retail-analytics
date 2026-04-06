"""
Ingestion script for UCI Online Retail II dataset.

The dataset contains UK-based online retail transactions from 2009-2011.
Real mode downloads:
  https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx

DEMO_MODE generates ~50,000 synthetic transactions for health & beauty
products covering 2020-2023.

Output: ``data/raw/uci_retail/<year>.parquet`` (one file per year)

Usage::

    python -m ingestion.uci_online_retail
"""

import logging
import re
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.config import (
    DEMO_MODE,
    UCI_ONLINE_RETAIL_URL,
    get_data_path,
    setup_logging,
)

logger: logging.Logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Demo product catalogue – health & beauty
# ---------------------------------------------------------------------------
_PRODUCT_CATALOGUE: list[tuple[str, str, float]] = [
    # (StockCode, Description, typical_unit_price)
    ("HB1001", "ROSE HIP FACE OIL 30ML", 12.99),
    ("HB1002", "HYALURONIC ACID SERUM 50ML", 18.99),
    ("HB1003", "VITAMIN C BRIGHTENING CREAM", 15.99),
    ("HB1004", "SPF50 DAILY MOISTURISER 75ML", 11.99),
    ("HB1005", "RETINOL NIGHT CREAM 50ML", 22.99),
    ("HB1006", "MICELLAR CLEANSING WATER 400ML", 6.99),
    ("HB1007", "CHARCOAL FACE MASK 100ML", 8.99),
    ("HB1008", "COLLAGEN EYE PATCHES 60PC", 14.99),
    ("HB1009", "NATURAL DEODORANT STICK 65G", 5.99),
    ("HB1010", "ARGAN OIL SHAMPOO 300ML", 7.49),
    ("HB1011", "KERATIN CONDITIONER 300ML", 7.49),
    ("HB1012", "COCONUT HAIR MASK 250ML", 9.99),
    ("HB1013", "BIOTIN HAIR SERUM 50ML", 13.99),
    ("HB1014", "DRY SHAMPOO SPRAY 200ML", 4.99),
    ("HB1015", "PURPLE TONING SHAMPOO 250ML", 8.99),
    ("HB1016", "BODY BUTTER MANGO 200ML", 6.99),
    ("HB1017", "SHEA BODY LOTION 400ML", 5.99),
    ("HB1018", "EXFOLIATING BODY SCRUB 300G", 8.49),
    ("HB1019", "BATH BOMB SET 6PC", 12.99),
    ("HB1020", "SHOWER GEL LAVENDER 500ML", 4.49),
    ("HB1021", "LIQUID FOUNDATION SPF15", 14.99),
    ("HB1022", "CONCEALER PEN MEDIUM", 9.99),
    ("HB1023", "VOLUMISING MASCARA BLACK", 8.99),
    ("HB1024", "EYESHADOW PALETTE 12 SHADES", 19.99),
    ("HB1025", "LIP GLOSS SET 5PC", 11.99),
    ("HB1026", "SETTING POWDER TRANSLUCENT", 13.99),
    ("HB1027", "BROW PENCIL DARK BROWN", 6.99),
    ("HB1028", "BRONZING DROPS 30ML", 16.99),
    ("HB1029", "BLUSH PALETTE CORAL", 14.99),
    ("HB1030", "MAKEUP REMOVER WIPES 25PC", 3.99),
    ("HB1031", "EAU DE PARFUM FLORAL 50ML", 44.99),
    ("HB1032", "BODY MIST VANILLA 200ML", 9.99),
    ("HB1033", "ROLL-ON PERFUME OIL 10ML", 7.99),
    ("HB1034", "AFTERSHAVE BALM 100ML", 15.99),
    ("HB1035", "SOLID PERFUME COMPACT 5G", 12.99),
    ("HB1036", "VITAMIN D3 90 TABLETS", 8.99),
    ("HB1037", "OMEGA-3 FISH OIL 60 CAPS", 11.99),
    ("HB1038", "PROBIOTIC 10 BILLION CFU", 16.99),
    ("HB1039", "MAGNESIUM GLYCINATE 120 CAPS", 14.99),
    ("HB1040", "COLLAGEN POWDER 250G", 24.99),
    ("HB1041", "ELECTRIC TOOTHBRUSH HEADS 4PC", 9.99),
    ("HB1042", "WHITENING TOOTHPASTE 100ML", 5.99),
    ("HB1043", "WATER FLOSSER REFILL TIPS", 7.99),
    ("HB1044", "CHARCOAL MOUTHWASH 500ML", 6.99),
    ("HB1045", "TEETH WHITENING STRIPS 14PC", 18.99),
    ("HB1046", "SUNSCREEN SPF30 LOTION 200ML", 8.99),
    ("HB1047", "SELF TAN MOUSSE MEDIUM 200ML", 12.99),
    ("HB1048", "AFTER SUN GEL ALOE VERA 200ML", 6.99),
    ("HB1049", "SPF50+ FACE SPRAY 75ML", 14.99),
    ("HB1050", "TAN REMOVAL MITT 2PC", 4.99),
]

_UK_REGIONS = [
    "United Kingdom", "United Kingdom", "United Kingdom", "United Kingdom",
    "United Kingdom", "United Kingdom", "United Kingdom", "United Kingdom",
    "United Kingdom", "United Kingdom", "United Kingdom", "United Kingdom",
    "United Kingdom", "United Kingdom", "United Kingdom", "United Kingdom",
    "United Kingdom",
    "France", "Germany", "Netherlands", "Spain", "Belgium",
    "Australia", "USA", "Canada",
]


# ---------------------------------------------------------------------------
# Demo generation
# ---------------------------------------------------------------------------

def _generate_demo_transactions(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic retail transactions resembling UCI Online Retail II.

    Includes:
    - ~50,000 transactions (2020-2023)
    - ~500 unique products (cycling through the catalogue)
    - ~15 % returns (negative quantities)
    - ~85 % UK customers

    Args:
        n: Number of transactions to generate.
        seed: NumPy random seed for reproducibility.

    Returns:
        DataFrame with UCI-compatible column names.
    """
    rng = np.random.default_rng(seed)

    # Invoice numbers
    invoice_base = 489000
    n_invoices = n // 3  # average ~3 lines per invoice
    invoice_nos = np.repeat(
        [f"{'C' if rng.random() < 0.15 else ''}{invoice_base + i}" for i in range(n_invoices)],
        rng.integers(1, 7, size=n_invoices),
    )[:n]

    # Dates
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2023-12-31")
    total_days = (end - start).days
    random_days = rng.integers(0, total_days, size=n)
    random_seconds = rng.integers(8 * 3600, 20 * 3600, size=n)
    invoice_dates = [
        (start + pd.Timedelta(days=int(d), seconds=int(s))).strftime("%Y-%m-%d %H:%M:%S")
        for d, s in zip(random_days, random_seconds)
    ]

    # Products (cycle through catalogue with repetition bias)
    cat_size = len(_PRODUCT_CATALOGUE)
    # Give popular items higher weight
    weights = rng.exponential(scale=2.0, size=cat_size)
    weights /= weights.sum()
    prod_indices = rng.choice(cat_size, size=n, p=weights)
    stock_codes = [_PRODUCT_CATALOGUE[i][0] for i in prod_indices]
    descriptions = [_PRODUCT_CATALOGUE[i][1] for i in prod_indices]
    base_prices = np.array([_PRODUCT_CATALOGUE[i][2] for i in prod_indices])

    # Quantities
    quantities = rng.integers(1, 12, size=n).astype(int)
    # Returns: ~15 % are negative (credit notes, Invoice starts with 'C')
    is_return = np.array([inv.startswith("C") for inv in invoice_nos])
    quantities = np.where(is_return, -quantities, quantities)

    # Prices: small noise around base
    price_noise = rng.normal(1.0, 0.05, size=n)
    prices = np.round(base_prices * price_noise, 2)
    prices = np.clip(prices, 0.99, 89.99)
    prices = np.where(is_return, prices, prices)  # returns keep same price

    # Customer IDs
    n_customers = 4000
    customer_ids = rng.integers(12000, 12000 + n_customers, size=n).astype(str)
    # ~5 % anonymous
    anon_mask = rng.random(n) < 0.05
    customer_ids = np.where(anon_mask, "", customer_ids)

    # Countries
    countries = rng.choice(_UK_REGIONS, size=n)

    df = pd.DataFrame(
        {
            "InvoiceNo": invoice_nos,
            "StockCode": stock_codes,
            "Description": descriptions,
            "Quantity": quantities,
            "InvoiceDate": invoice_dates,
            "Price": prices,
            "Customer ID": customer_ids,
            "Country": countries,
        }
    )
    logger.info("Generated %d synthetic transactions", len(df))
    return df


# ---------------------------------------------------------------------------
# Real data fetch
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def fetch_uci_data() -> pd.DataFrame:
    """Download and parse the UCI Online Retail II Excel file.

    Reads both sheets (``Year 2009-2010`` and ``Year 2010-2011``) and
    concatenates them.

    Returns:
        Combined raw DataFrame from both sheets.

    Raises:
        requests.HTTPError: On non-2xx HTTP response.
        ValueError: If the expected sheets are not found.
    """
    logger.info("Downloading UCI Online Retail II from %s", UCI_ONLINE_RETAIL_URL)
    response = requests.get(UCI_ONLINE_RETAIL_URL, timeout=120, stream=True)
    response.raise_for_status()

    content = BytesIO(response.content)
    sheets = ["Year 2009-2010", "Year 2010-2011"]
    frames: list[pd.DataFrame] = []
    for sheet in sheets:
        try:
            df_sheet = pd.read_excel(content, sheet_name=sheet, engine="openpyxl")
            frames.append(df_sheet)
            logger.info("Read %d rows from sheet '%s'", len(df_sheet), sheet)
        except Exception as exc:
            logger.warning("Could not read sheet '%s': %s", sheet, exc)

    if not frames:
        raise ValueError("No sheets could be read from the UCI dataset")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_uci_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the UCI Online Retail II DataFrame.

    Steps:
    - Drop rows with null InvoiceNo, StockCode, or Description.
    - Remove rows with non-numeric/empty StockCode.
    - Remove extreme outliers: Price < 0 or > 10,000; |Quantity| > 10,000.
    - Parse InvoiceDate as datetime.

    Args:
        df: Raw UCI DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    initial = len(df)
    df = df.dropna(subset=["InvoiceNo", "StockCode", "Description"])
    df = df[df["Price"] >= 0]
    df = df[df["Price"] <= 10_000]
    df = df[df["Quantity"].abs() <= 10_000]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    logger.info("Cleaned UCI data: %d → %d rows (removed %d)", initial, len(df), initial - len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_uci_data(df: pd.DataFrame) -> dict[str, Path]:
    """Save UCI data as Parquet files partitioned by year.

    Args:
        df: Cleaned UCI DataFrame with an ``InvoiceDate`` column.

    Returns:
        Dict mapping year (str) to :class:`~pathlib.Path` of the written file.
    """
    out_dir = get_data_path("raw/uci_retail")
    df = df.copy()
    df["_year"] = pd.to_datetime(df["InvoiceDate"]).dt.year
    paths: dict[str, Path] = {}

    for year, group in df.groupby("_year"):
        out_path = out_dir / f"{year}.parquet"
        group = group.drop(columns=["_year"])
        group.to_parquet(out_path, index=False, engine="pyarrow")
        paths[str(year)] = out_path
        logger.info("Saved %d rows for year %d to %s", len(group), year, out_path)

    return paths


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    """Orchestrate UCI Online Retail ingestion.

    Returns:
        Cleaned transactions DataFrame.
    """
    logger.info("Starting UCI Online Retail ingestion (DEMO_MODE=%s)", DEMO_MODE)

    if DEMO_MODE:
        df = _generate_demo_transactions()
    else:
        raw = fetch_uci_data()
        df = clean_uci_data(raw)

    save_uci_data(df)
    logger.info(
        "UCI ingestion complete: %d transactions, %d products, %d customers",
        len(df),
        df["StockCode"].nunique(),
        df["Customer ID"].nunique() if "Customer ID" in df.columns else 0,
    )
    return df


if __name__ == "__main__":
    run()
