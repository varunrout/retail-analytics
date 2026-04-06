"""
Ingestion script for Open-Meteo weather data.

Source: https://archive-api.open-meteo.com/v1/archive
Fetches daily weather for UK cities (London, Manchester, Birmingham)
and saves NDJSON files partitioned by year-month under
``data/raw/weather/<location>/``.

Usage::

    python -m ingestion.open_meteo_weather
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.config import (
    DEMO_MODE,
    OPEN_METEO_ARCHIVE_URL,
    UK_WEATHER_LOCATIONS,
    get_data_path,
    setup_logging,
)

logger: logging.Logger = setup_logging(__name__)

# WMO weather interpretation codes (subset used in demo generation)
_SUNNY_CODES = [0, 1]        # clear sky
_CLOUDY_CODES = [2, 3, 45]   # partly cloudy / overcast / fog
_RAIN_CODES = [51, 61, 63, 80, 81]  # drizzle / rain / showers
_SNOW_CODES = [71, 73, 85]   # snow


# ---------------------------------------------------------------------------
# Demo data generation
# ---------------------------------------------------------------------------

def _generate_daily_weather(
    location: str,
    start_date: str,
    end_date: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate synthetic daily weather data with realistic seasonal patterns.

    Temperature and precipitation follow sinusoidal seasonal curves:
    - Temperature peaks in July/August (summer) and troughs in January.
    - Precipitation is higher in autumn/winter months.

    Args:
        location: City name used for slight per-city offsets.
        start_date: ISO date string ``YYYY-MM-DD``.
        end_date: ISO date string ``YYYY-MM-DD``.
        seed: NumPy random seed for reproducibility.

    Returns:
        Dict matching the Open-Meteo archive API response structure.
    """
    rng = np.random.default_rng(seed + hash(location) % 1000)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    # Day-of-year fraction in [0, 2π]
    doy = np.array([d.timetuple().tm_yday for d in dates])
    phase = 2 * np.pi * doy / 365.0

    # --- Temperature --------------------------------------------------------
    # UK mean: ~12 °C; amplitude ~8 °C; peak around day ~200 (mid July)
    city_offset = {"London": 1.5, "Manchester": -0.5, "Birmingham": 0.0}.get(location, 0.0)
    temp_mean_c = 12.0 + city_offset - 8.0 * np.cos(phase)
    temp_max_c = temp_mean_c + rng.uniform(2.0, 6.0, n)
    temp_min_c = temp_mean_c - rng.uniform(2.0, 6.0, n)

    # --- Precipitation ------------------------------------------------------
    # Higher in autumn/winter; base ~1.5 mm/day, amplitude ~0.8 mm
    precip_base = 1.5 + 0.8 * np.cos(phase)  # more in winter
    precip_mm = np.clip(rng.exponential(scale=precip_base, size=n), 0, 40)

    # --- Weather codes ------------------------------------------------------
    weather_codes = []
    for precip, tmax in zip(precip_mm, temp_max_c):
        if precip < 0.1:
            code = rng.choice(_SUNNY_CODES)
        elif precip < 2.0:
            code = rng.choice(_CLOUDY_CODES)
        elif tmax < 2.0:
            code = rng.choice(_SNOW_CODES)
        else:
            code = rng.choice(_RAIN_CODES)
        weather_codes.append(int(code))

    return {
        "daily": {
            "time": [d.strftime("%Y-%m-%d") for d in dates],
            "temperature_2m_max": temp_max_c.round(1).tolist(),
            "temperature_2m_min": temp_min_c.round(1).tolist(),
            "precipitation_sum": precip_mm.round(1).tolist(),
            "weathercode": weather_codes,
        },
        "timezone": "Europe/London",
    }


# ---------------------------------------------------------------------------
# Real API fetch
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_weather(lat: float, lon: float, start_date: str, end_date: str) -> dict[str, Any]:
    """Fetch daily weather archive from Open-Meteo.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        start_date: ISO date string ``YYYY-MM-DD``.
        end_date: ISO date string ``YYYY-MM-DD``.

    Returns:
        Parsed JSON response from the Open-Meteo archive API.

    Raises:
        requests.HTTPError: On non-2xx response.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "weathercode"],
        "timezone": "Europe/London",
    }
    logger.info("Fetching weather lat=%s lon=%s %s→%s", lat, lon, start_date, end_date)
    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_weather(raw: dict[str, Any], location: str) -> pd.DataFrame:
    """Convert a raw Open-Meteo response dict into a tidy DataFrame.

    Args:
        raw: Dict as returned by :func:`fetch_weather` (or the demo generator).
        location: Human-readable city name to add as a column.

    Returns:
        DataFrame with columns: ``date``, ``location``, ``temp_max_c``,
        ``temp_min_c``, ``precipitation_mm``, ``weather_code``.
    """
    daily = raw["daily"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]).date,
            "location": location,
            "temp_max_c": daily["temperature_2m_max"],
            "temp_min_c": daily["temperature_2m_min"],
            "precipitation_mm": daily["precipitation_sum"],
            "weather_code": daily["weathercode"],
        }
    )
    logger.debug("Parsed %d weather rows for %s", len(df), location)
    return df


def add_season_label(df: pd.DataFrame) -> pd.DataFrame:
    """Append a ``season`` column based on calendar month (UK meteorological seasons).

    Mapping:
        - Spring: March–May
        - Summer: June–August
        - Autumn: September–November
        - Winter: December–February

    Args:
        df: DataFrame with a ``date`` column (``datetime.date`` or parseable).

    Returns:
        Input DataFrame with an additional ``season`` string column.
    """
    months = pd.to_datetime(df["date"]).dt.month

    def _season(m: int) -> str:
        if m in (3, 4, 5):
            return "Spring"
        if m in (6, 7, 8):
            return "Summer"
        if m in (9, 10, 11):
            return "Autumn"
        return "Winter"

    df = df.copy()
    df["season"] = months.map(_season)
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_weather(df: pd.DataFrame) -> list[Path]:
    """Save weather data as NDJSON files partitioned by location and year-month.

    Output pattern::

        data/raw/weather/<location>/<YYYY-MM>.json

    Args:
        df: Weather DataFrame containing at least ``date`` and ``location`` columns.

    Returns:
        List of :class:`~pathlib.Path` objects for every file written.
    """
    written: list[Path] = []
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"])
    df["_ym"] = df["_date"].dt.strftime("%Y-%m")

    for (location, ym), group in df.groupby(["location", "_ym"]):
        out_dir = get_data_path(f"raw/weather/{location.lower()}")
        out_path = out_dir / f"{ym}.json"
        group = group.drop(columns=["_date", "_ym"])
        group["date"] = group["date"].astype(str)
        with out_path.open("w", encoding="utf-8") as fh:
            for record in group.to_dict(orient="records"):
                fh.write(json.dumps(record) + "\n")
        written.append(out_path)

    logger.info("Saved %d partition files for weather data", len(written))
    return written


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _missing_months(location: str, start_date: str) -> list[tuple[str, str]]:
    """Return (start, end) date pairs for months not yet present on disk.

    Args:
        location: City name (lowercase directory).
        start_date: ISO date string to start from if no files exist.

    Returns:
        List of ``(start, end)`` ISO date strings for each missing month.
    """
    location_dir = get_data_path(f"raw/weather/{location.lower()}")
    existing = {p.stem for p in location_dir.glob("*.json")}

    today = date.today()
    current = datetime.strptime(start_date, "%Y-%m-%d").date().replace(day=1)
    # Only fetch up to the end of last month (archive data)
    last_available = (today.replace(day=1) - timedelta(days=1)).replace(day=1)

    missing: list[tuple[str, str]] = []
    while current <= last_available:
        ym = current.strftime("%Y-%m")
        if ym not in existing:
            next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
            month_end = min(next_month - timedelta(days=1), last_available)
            missing.append((current.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")))
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)

    return missing


def run(start_date: str = "2020-01-01", incremental: bool = True) -> pd.DataFrame:
    """Orchestrate weather ingestion for all configured UK locations.

    In ``DEMO_MODE`` synthetic data is generated without HTTP calls.
    When ``incremental=True`` only missing months are fetched/generated.

    Args:
        start_date: ISO date string for the historical start of the pull.
        incremental: Skip months that already have files on disk.

    Returns:
        Combined DataFrame for all locations.
    """
    logger.info(
        "Starting weather ingestion (DEMO_MODE=%s, incremental=%s, start=%s)",
        DEMO_MODE,
        incremental,
        start_date,
    )
    frames: list[pd.DataFrame] = []

    seed_counter = 0
    for location, coords in UK_WEATHER_LOCATIONS.items():
        if incremental:
            missing = _missing_months(location, start_date)
        else:
            today = date.today()
            last_month_end = (today.replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
            missing = [(start_date, last_month_end)]

        if not missing:
            logger.info("No missing months for %s – skipping", location)
            continue

        logger.info("Fetching %d month-range(s) for %s", len(missing), location)
        for s, e in missing:
            if DEMO_MODE:
                raw = _generate_daily_weather(location, s, e, seed=seed_counter)
            else:
                raw = fetch_weather(coords["lat"], coords["lon"], s, e)
            seed_counter += 1
            df_loc = parse_weather(raw, location)
            df_loc = add_season_label(df_loc)
            frames.append(df_loc)

    if not frames:
        logger.warning("No new weather data collected.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    save_weather(combined)
    logger.info(
        "Weather ingestion complete: %d rows across %d locations",
        len(combined),
        combined["location"].nunique(),
    )
    return combined


if __name__ == "__main__":
    run()
