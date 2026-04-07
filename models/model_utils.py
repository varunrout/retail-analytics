"""Shared utilities for model training, scoring, and artifact persistence."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"


def ensure_output_dirs() -> dict[str, Path]:
    directories = {
        "data": DATA_DIR,
        "synthetic": SYNTHETIC_DIR,
        "features": FEATURES_DIR,
        "models": MODELS_DIR,
        "reports": REPORTS_DIR,
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_serialize),
        encoding="utf-8",
    )
    return path


def save_pickle(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def standardize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    standardized = df.copy()
    if "stock_code" not in standardized.columns and "sku_id" in standardized.columns:
        standardized = standardized.rename(columns={"sku_id": "stock_code"})
    if "sku" not in standardized.columns and "stock_code" in standardized.columns:
        standardized["sku"] = standardized["stock_code"]
    if "category_l1" not in standardized.columns and "category" in standardized.columns:
        standardized["category_l1"] = standardized["category"]
    if "invoice_id" not in standardized.columns and "invoice_no" in standardized.columns:
        standardized["invoice_id"] = standardized["invoice_no"]
    if "invoice_date" in standardized.columns:
        standardized["invoice_date"] = pd.to_datetime(standardized["invoice_date"])
    return standardized


def standardize_inventory(df: pd.DataFrame) -> pd.DataFrame:
    standardized = df.copy()
    if "stock_code" not in standardized.columns and "sku_id" in standardized.columns:
        standardized = standardized.rename(columns={"sku_id": "stock_code"})
    if "sku" not in standardized.columns and "stock_code" in standardized.columns:
        standardized["sku"] = standardized["stock_code"]
    if "category_l1" not in standardized.columns and "category" in standardized.columns:
        standardized["category_l1"] = standardized["category"]
    return standardized


def standardize_costs(df: pd.DataFrame) -> pd.DataFrame:
    standardized = df.copy()
    if "stock_code" not in standardized.columns and "sku_id" in standardized.columns:
        standardized = standardized.rename(columns={"sku_id": "stock_code"})
    if "sku" not in standardized.columns and "stock_code" in standardized.columns:
        standardized["sku"] = standardized["stock_code"]
    return standardized


def standardize_customers(df: pd.DataFrame) -> pd.DataFrame:
    standardized = df.copy()
    for col in ["first_purchase_date", "last_purchase_date"]:
        if col in standardized.columns:
            standardized[col] = pd.to_datetime(standardized[col])
    return standardized


def make_demo_transactions(n: int = 5_000) -> pd.DataFrame:
    from synthetic.generate_transactions import generate_base_transactions

    return standardize_transactions(generate_base_transactions(n=n))


def load_input_tables(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    ensure_output_dirs()

    required_paths = {
        "transactions": SYNTHETIC_DIR / "transactions.parquet",
        "inventory": SYNTHETIC_DIR / "inventory.parquet",
        "customers": SYNTHETIC_DIR / "customers.parquet",
        "campaigns": SYNTHETIC_DIR / "campaigns.parquet",
        "costs": SYNTHETIC_DIR / "costs.parquet",
    }

    if force_refresh or any(not path.exists() for path in required_paths.values()):
        from synthetic.seed_all import run_all

        logger.info("Generating synthetic inputs (force_refresh=%s)", force_refresh)
        run_all(force_refresh=force_refresh)

    tables: dict[str, pd.DataFrame] = {}
    for name, path in required_paths.items():
        if not path.exists():
            logger.warning("Synthetic table missing: %s", path)
            continue
        tables[name] = pd.read_parquet(path)

    if "transactions" in tables:
        tables["transactions"] = standardize_transactions(tables["transactions"])
    if "inventory" in tables:
        tables["inventory"] = standardize_inventory(tables["inventory"])
    if "costs" in tables:
        tables["costs"] = standardize_costs(tables["costs"])
    if "customers" in tables:
        tables["customers"] = standardize_customers(tables["customers"])
    return tables


def compute_mape(actual: pd.Series, predicted: pd.Series) -> float:
    actual_clean = pd.Series(actual).astype(float)
    predicted_clean = pd.Series(predicted).astype(float)
    mask = actual_clean.ne(0) & actual_clean.notna() & predicted_clean.notna()
    if not mask.any():
        return 0.0
    return float((actual_clean[mask] - predicted_clean[mask]).abs().div(actual_clean[mask]).mean())


def risk_band_from_probability(value: float) -> str:
    if value >= 0.7:
        return "High"
    if value >= 0.4:
        return "Medium"
    return "Low"


def latest_file(path_pattern: str) -> Path | None:
    matches = sorted(MODELS_DIR.glob(path_pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None