"""
Ingestion script for Google Trends data.

Uses pytrends to pull weekly relative interest (0-100) for health & beauty
keywords in the UK. In DEMO_MODE realistic synthetic series are generated
using sinusoidal seasonal patterns.

Keywords:
  skincare, moisturiser, sunscreen, vitamin supplements,
  hair dye, perfume, lip balm

Output: ``data/raw/google_trends/trends.json`` (NDJSON)

Usage::

    python -m ingestion.google_trends
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.config import (
    DEMO_MODE,
    UK_HEALTH_BEAUTY_KEYWORDS,
    get_data_path,
    setup_logging,
)

logger: logging.Logger = setup_logging(__name__)

# Keywords to track (subset of UK_HEALTH_BEAUTY_KEYWORDS)
_TREND_KEYWORDS: list[str] = [
    "skincare",
    "moisturiser",
    "sunscreen",
    "vitamin supplements",
    "hair dye",
    "perfume",
    "lip balm",
]

# ---------------------------------------------------------------------------
# Seasonal peak parameters  (month_peak: 1-12, amplitude: 0-1, base: 0-100)
# Each tuple: (month_peak, amplitude, base_level, long_term_cagr)
# ---------------------------------------------------------------------------
_KEYWORD_PARAMS: dict[str, tuple[float, float, float, float]] = {
    "skincare":            (10.0, 12.0, 55.0, 0.08),   # peak autumn, strong growth
    "moisturiser":         (11.5, 18.0, 42.0, 0.04),   # peak Oct-Feb
    "sunscreen":           (6.5,  30.0, 25.0, 0.06),   # peak May-Aug
    "vitamin supplements": (1.5,  20.0, 35.0, 0.03),   # peak Jan & Sep
    "hair dye":            (3.5,  10.0, 40.0, 0.01),   # mild spring peak
    "perfume":             (11.5, 22.0, 35.0, 0.02),   # peak Christmas
    "lip balm":            (1.0,  15.0, 30.0, 0.02),   # peak winter
}


# ---------------------------------------------------------------------------
# Demo data generation
# ---------------------------------------------------------------------------

def _generate_demo_trends(
    start_date: str = "2019-01-06",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic weekly Google Trends data with seasonal patterns.

    Each keyword is modelled as:

    .. code-block:: text

        trend(t) = base + long_term_growth(t)
                   + amplitude * sin(2π*(week_of_year - peak_week) / 52)
                   + noise

    Results are normalised per-keyword to the range [0, 100].

    Args:
        start_date: ISO Monday date for the first weekly observation.
        seed: NumPy random seed for reproducibility.

    Returns:
        Long-format DataFrame with columns: ``date``, ``keyword``,
        ``relative_interest``.
    """
    rng = np.random.default_rng(seed)

    weeks = pd.date_range(start=start_date, end=pd.Timestamp.now(), freq="W-SUN")
    n = len(weeks)
    week_of_year = weeks.isocalendar().week.to_numpy().astype(float)
    year_frac = (weeks.year - weeks.year.min()) / max(1, weeks.year.max() - weeks.year.min())

    records: list[dict] = []
    for keyword in _TREND_KEYWORDS:
        peak_month, amplitude, base, cagr = _KEYWORD_PARAMS.get(
            keyword, (6.0, 10.0, 50.0, 0.02)
        )
        peak_week = peak_month * (52.0 / 12.0)

        # Dual-peak for vitamin supplements (January + September)
        if keyword == "vitamin supplements":
            seasonal = (
                amplitude * np.sin(2 * np.pi * (week_of_year - 1) / 52)
                + (amplitude * 0.6) * np.sin(2 * np.pi * (week_of_year - 36) / 52)
            )
        else:
            seasonal = amplitude * np.sin(2 * np.pi * (week_of_year - peak_week) / 52)

        long_term = base * ((1 + cagr) ** (year_frac * (weeks.year.max() - weeks.year.min())))
        noise = rng.normal(0, 2.5, n)
        raw_series = long_term + seasonal + noise

        # Normalise to [0, 100]
        series_min = raw_series.min()
        series_max = raw_series.max()
        if series_max > series_min:
            normalised = (raw_series - series_min) / (series_max - series_min) * 100
        else:
            normalised = np.full(n, 50.0)
        normalised = np.clip(normalised, 0, 100)

        for dt, val in zip(weeks, normalised):
            records.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "keyword": keyword,
                    "relative_interest": round(float(val), 1),
                }
            )

    df = pd.DataFrame(records)
    logger.info(
        "Generated %d synthetic trend rows (%d weeks × %d keywords)",
        len(df),
        n,
        len(_TREND_KEYWORDS),
    )
    return df


# ---------------------------------------------------------------------------
# Real pytrends fetch
# ---------------------------------------------------------------------------

def fetch_trends(
    keywords: list[str] | None = None,
    timeframe: str = "today 5-y",
    geo: str = "GB",
) -> pd.DataFrame:
    """Fetch weekly Google Trends data using pytrends.

    pytrends is imported lazily so the rest of the module works without it
    installed (DEMO_MODE doesn't require it).

    Args:
        keywords: List of search terms. Defaults to :data:`_TREND_KEYWORDS`.
        timeframe: pytrends timeframe string (default ``"today 5-y"``).
        geo: Geographic area code (default ``"GB"`` for United Kingdom).

    Returns:
        Raw wide-format DataFrame from pytrends (columns = keywords, index = date).

    Raises:
        ImportError: If pytrends is not installed.
        Exception: On API errors from Google.
    """
    if keywords is None:
        keywords = _TREND_KEYWORDS

    try:
        from pytrends.request import TrendReq  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pytrends is required for real mode: pip install pytrends"
        ) from exc

    pytrends = TrendReq(hl="en-GB", tz=0, timeout=(10, 25))

    # pytrends supports max 5 keywords per batch
    frames: list[pd.DataFrame] = []
    for i in range(0, len(keywords), 5):
        batch = keywords[i : i + 5]
        pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo)
        df_batch = pytrends.interest_over_time()
        if not df_batch.empty:
            if "isPartial" in df_batch.columns:
                df_batch = df_batch.drop(columns=["isPartial"])
            frames.append(df_batch)
        time.sleep(1.5)  # courtesy pause to avoid rate-limiting

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_trends(raw: pd.DataFrame) -> pd.DataFrame:
    """Melt a wide-format pytrends DataFrame to long format.

    Args:
        raw: Wide DataFrame with date index and keyword columns.

    Returns:
        Long-format DataFrame with columns: ``date``, ``keyword``,
        ``relative_interest``.
    """
    df = raw.reset_index().rename(columns={"date": "date"})
    if "date" not in df.columns and df.index.name == "date":
        df = raw.reset_index()

    long_df = df.melt(id_vars=["date"], var_name="keyword", value_name="relative_interest")
    long_df["date"] = pd.to_datetime(long_df["date"]).dt.strftime("%Y-%m-%d")
    long_df["relative_interest"] = long_df["relative_interest"].round(1)
    long_df = long_df.dropna(subset=["relative_interest"])
    long_df = long_df.sort_values(["keyword", "date"]).reset_index(drop=True)
    logger.debug("Parsed %d trend rows from wide format", len(long_df))
    return long_df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_trends(df: pd.DataFrame) -> Path:
    """Save Google Trends data to NDJSON.

    Args:
        df: Long-format trends DataFrame.

    Returns:
        :class:`~pathlib.Path` of the written file.
    """
    out_dir = get_data_path("raw/google_trends")
    out_path = out_dir / "trends.json"
    with out_path.open("w", encoding="utf-8") as fh:
        for record in df.to_dict(orient="records"):
            fh.write(json.dumps(record) + "\n")
    logger.info("Saved %d trend records to %s", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(keywords: list[str] | None = None) -> pd.DataFrame:
    """Orchestrate Google Trends ingestion.

    In DEMO_MODE synthetic series are generated. In real mode pytrends is used.

    Args:
        keywords: Override the default keyword list.

    Returns:
        Long-format trends DataFrame.
    """
    if keywords is None:
        keywords = _TREND_KEYWORDS

    logger.info("Starting Google Trends ingestion (DEMO_MODE=%s)", DEMO_MODE)

    if DEMO_MODE:
        df = _generate_demo_trends()
    else:
        raw = fetch_trends(keywords=keywords)
        if raw.empty:
            logger.warning("No trends data returned – using demo data")
            df = _generate_demo_trends()
        else:
            df = parse_trends(raw)

    save_trends(df)
    logger.info(
        "Trends ingestion complete: %d rows, %d keywords",
        len(df),
        df["keyword"].nunique(),
    )
    return df


if __name__ == "__main__":
    run()
