"""
Ingestion script for UK Bank Holidays.

Source: https://www.gov.uk/bank-holidays.json
Fetches England & Wales bank holidays, returns a tidy DataFrame, and saves
the result to ``data/raw/bank_holidays/bank_holidays.json`` in NDJSON format.

Usage::

    python -m ingestion.uk_bank_holidays
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.config import (
    BANK_HOLIDAYS_URL,
    DEMO_MODE,
    RAW_BANK_HOLIDAYS_DIR,
    get_data_path,
    setup_logging,
)

logger: logging.Logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Demo data – 59 known UK (England & Wales) bank holidays 2020-2026
# ---------------------------------------------------------------------------
_DEMO_HOLIDAYS: list[dict] = [
    # 2020
    {"date": "2020-01-01", "name": "New Year's Day"},
    {"date": "2020-04-10", "name": "Good Friday"},
    {"date": "2020-04-13", "name": "Easter Monday"},
    {"date": "2020-05-08", "name": "Early May bank holiday (VE day)"},
    {"date": "2020-05-25", "name": "Spring bank holiday"},
    {"date": "2020-08-31", "name": "Summer bank holiday"},
    {"date": "2020-12-25", "name": "Christmas Day"},
    {"date": "2020-12-28", "name": "Boxing Day (substitute day)"},
    # 2021
    {"date": "2021-01-01", "name": "New Year's Day"},
    {"date": "2021-04-02", "name": "Good Friday"},
    {"date": "2021-04-05", "name": "Easter Monday"},
    {"date": "2021-05-03", "name": "Early May bank holiday"},
    {"date": "2021-05-31", "name": "Spring bank holiday"},
    {"date": "2021-08-30", "name": "Summer bank holiday"},
    {"date": "2021-12-27", "name": "Christmas Day (substitute day)"},
    {"date": "2021-12-28", "name": "Boxing Day (substitute day)"},
    # 2022
    {"date": "2022-01-03", "name": "New Year's Day (substitute day)"},
    {"date": "2022-04-15", "name": "Good Friday"},
    {"date": "2022-04-18", "name": "Easter Monday"},
    {"date": "2022-05-02", "name": "Early May bank holiday"},
    {"date": "2022-06-02", "name": "Spring bank holiday"},
    {"date": "2022-06-03", "name": "Platinum Jubilee bank holiday"},
    {"date": "2022-08-29", "name": "Summer bank holiday"},
    {"date": "2022-09-19", "name": "Bank Holiday for the State Funeral of Queen Elizabeth II"},
    {"date": "2022-12-26", "name": "Boxing Day"},
    {"date": "2022-12-27", "name": "Christmas Day (substitute day)"},
    # 2023
    {"date": "2023-01-02", "name": "New Year's Day (substitute day)"},
    {"date": "2023-04-07", "name": "Good Friday"},
    {"date": "2023-04-10", "name": "Easter Monday"},
    {"date": "2023-05-01", "name": "Early May bank holiday"},
    {"date": "2023-05-08", "name": "Bank holiday for the coronation of King Charles III"},
    {"date": "2023-05-29", "name": "Spring bank holiday"},
    {"date": "2023-08-28", "name": "Summer bank holiday"},
    {"date": "2023-12-25", "name": "Christmas Day"},
    {"date": "2023-12-26", "name": "Boxing Day"},
    # 2024
    {"date": "2024-01-01", "name": "New Year's Day"},
    {"date": "2024-03-29", "name": "Good Friday"},
    {"date": "2024-04-01", "name": "Easter Monday"},
    {"date": "2024-05-06", "name": "Early May bank holiday"},
    {"date": "2024-05-27", "name": "Spring bank holiday"},
    {"date": "2024-08-26", "name": "Summer bank holiday"},
    {"date": "2024-12-25", "name": "Christmas Day"},
    {"date": "2024-12-26", "name": "Boxing Day"},
    # 2025
    {"date": "2025-01-01", "name": "New Year's Day"},
    {"date": "2025-04-18", "name": "Good Friday"},
    {"date": "2025-04-21", "name": "Easter Monday"},
    {"date": "2025-05-05", "name": "Early May bank holiday"},
    {"date": "2025-05-26", "name": "Spring bank holiday"},
    {"date": "2025-08-25", "name": "Summer bank holiday"},
    {"date": "2025-12-25", "name": "Christmas Day"},
    {"date": "2025-12-26", "name": "Boxing Day"},
    # 2026
    {"date": "2026-01-01", "name": "New Year's Day"},
    {"date": "2026-04-03", "name": "Good Friday"},
    {"date": "2026-04-06", "name": "Easter Monday"},
    {"date": "2026-05-04", "name": "Early May bank holiday"},
    {"date": "2026-05-25", "name": "Spring bank holiday"},
    {"date": "2026-08-31", "name": "Summer bank holiday"},
    {"date": "2026-12-25", "name": "Christmas Day"},
    {"date": "2026-12-28", "name": "Boxing Day (substitute day)"},
]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_bank_holidays() -> dict:
    """Fetch raw bank holidays JSON from the GOV.UK endpoint.

    Returns:
        Parsed JSON response as a :class:`dict` keyed by division name.

    Raises:
        requests.HTTPError: If the HTTP response indicates a server error.
        requests.ConnectionError: On network failure (retried up to 3 times).
    """
    logger.info("Fetching bank holidays from %s", BANK_HOLIDAYS_URL)
    response = requests.get(BANK_HOLIDAYS_URL, timeout=15)
    response.raise_for_status()
    logger.debug("Received %d bytes", len(response.content))
    return response.json()


def parse_bank_holidays(raw: dict) -> pd.DataFrame:
    """Parse the raw GOV.UK bank holidays payload into a tidy DataFrame.

    Args:
        raw: JSON dict as returned by :func:`fetch_bank_holidays`.

    Returns:
        DataFrame with columns:

        - ``date``  – :class:`datetime.date`
        - ``name``  – holiday title
        - ``region`` – GOV.UK division name (e.g. ``"england-and-wales"``)
        - ``year``  – calendar year (int)
        - ``month`` – calendar month 1-12 (int)
    """
    records: list[dict] = []
    for division, payload in raw.items():
        for event in payload.get("events", []):
            records.append(
                {
                    "date": event["date"],
                    "name": event["title"],
                    "region": division,
                }
            )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Parsed %d bank holiday records across %d divisions", len(df), df["region"].nunique())
    return df


def _demo_bank_holidays() -> pd.DataFrame:
    """Return a hardcoded DataFrame of 59 known UK bank holidays (demo mode)."""
    logger.info("DEMO_MODE: using hardcoded bank holiday data (%d entries)", len(_DEMO_HOLIDAYS))
    records = [
        {
            "date": h["date"],
            "name": h["name"],
            "region": "england-and-wales",
        }
        for h in _DEMO_HOLIDAYS
    ]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def save_bank_holidays(df: pd.DataFrame) -> Path:
    """Persist the bank holidays DataFrame to NDJSON.

    Args:
        df: Tidy bank holidays DataFrame as produced by :func:`parse_bank_holidays`.

    Returns:
        :class:`~pathlib.Path` of the written file.
    """
    output_dir = get_data_path("raw/bank_holidays")
    output_path = output_dir / "bank_holidays.json"

    # Convert date objects to ISO strings for JSON serialisation
    serialisable = df.copy()
    serialisable["date"] = serialisable["date"].astype(str)

    with output_path.open("w", encoding="utf-8") as fh:
        for record in serialisable.to_dict(orient="records"):
            fh.write(json.dumps(record) + "\n")

    logger.info("Saved %d records to %s", len(df), output_path)
    return output_path


def run() -> pd.DataFrame:
    """Orchestrate fetch → parse → save for UK bank holidays.

    In ``DEMO_MODE`` the HTTP call is skipped and hardcoded data is used instead.

    Returns:
        Tidy bank holidays DataFrame.
    """
    logger.info("Starting UK bank holidays ingestion (DEMO_MODE=%s)", DEMO_MODE)

    if DEMO_MODE:
        df = _demo_bank_holidays()
    else:
        raw = fetch_bank_holidays()
        df = parse_bank_holidays(raw)

    save_bank_holidays(df)
    logger.info(
        "Ingestion complete: %d records, years %s–%s",
        len(df),
        df["year"].min(),
        df["year"].max(),
    )
    return df


if __name__ == "__main__":
    run()
