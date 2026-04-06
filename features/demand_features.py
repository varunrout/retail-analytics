"""
Demand signal features for forecasting models.
"""
import pandas as pd
import numpy as np
from scipy import stats
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_rolling_demand(
    df: pd.DataFrame,
    value_col: str = "quantity",
    date_col: str = "invoice_date",
    group_col: str = "stock_code",
) -> pd.DataFrame:
    """
    Compute rolling demand features per SKU.

    Aggregates to weekly level first, then computes:
    - rolling_7d_units: 1-week rolling sum
    - rolling_28d_units: 4-week rolling sum
    - rolling_90d_units: 13-week rolling sum
    - rolling_28d_mean: 4-week rolling mean
    - rolling_28d_std: 4-week rolling std
    - rolling_28d_cv: coefficient of variation (std/mean)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_week"] = df[date_col].dt.to_period("W").dt.start_time

    weekly = (
        df.groupby([group_col, "_week"])[value_col]
        .sum()
        .reset_index()
        .rename(columns={"_week": "week_start"})
    )

    results = []
    for sku, grp in weekly.groupby(group_col):
        grp = grp.sort_values("week_start").set_index("week_start")
        s = grp[value_col]
        grp["rolling_7d_units"] = s.rolling(1, min_periods=1).sum()
        grp["rolling_28d_units"] = s.rolling(4, min_periods=1).sum()
        grp["rolling_90d_units"] = s.rolling(13, min_periods=1).sum()
        grp["rolling_28d_mean"] = s.rolling(4, min_periods=1).mean()
        grp["rolling_28d_std"] = s.rolling(4, min_periods=1).std().fillna(0)
        grp["rolling_28d_cv"] = grp["rolling_28d_std"] / grp["rolling_28d_mean"].replace(0, np.nan)
        grp[group_col] = sku
        grp = grp.reset_index()
        results.append(grp)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def compute_yoy_changes(
    df: pd.DataFrame,
    value_col: str,
    date_col: str,
    group_col: str,
) -> pd.DataFrame:
    """
    Compute year-over-year and month-over-month changes.
    Aligns by same calendar week (YoY) or same period (MoM).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_year"] = df[date_col].dt.year
    df["_month"] = df[date_col].dt.month
    df["_week"] = df[date_col].dt.isocalendar().week.astype(int)

    monthly = (
        df.groupby([group_col, "_year", "_month"])[value_col]
        .sum()
        .reset_index()
    )

    results = []
    for sku, grp in monthly.groupby(group_col):
        grp = grp.sort_values(["_year", "_month"]).copy()
        grp["mom_change"] = grp[value_col].pct_change()

        yoy_parts = []
        for _, row in grp.iterrows():
            prior = grp[
                (grp["_year"] == row["_year"] - 1) & (grp["_month"] == row["_month"])
            ]
            if not prior.empty:
                prior_val = prior[value_col].values[0]
                yoy = (row[value_col] - prior_val) / (prior_val if prior_val != 0 else np.nan)
            else:
                yoy = np.nan
            yoy_parts.append(yoy)

        grp["yoy_change"] = yoy_parts
        grp[group_col] = sku
        results.append(grp)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def compute_trend_slope(
    df: pd.DataFrame,
    value_col: str,
    date_col: str,
    group_col: str,
    window_weeks: int = 13,
) -> pd.DataFrame:
    """
    Compute linear trend slope over trailing window using OLS.
    Positive slope = growing demand. Normalize by mean to get relative trend.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_week"] = df[date_col].dt.to_period("W").dt.start_time

    weekly = (
        df.groupby([group_col, "_week"])[value_col]
        .sum()
        .reset_index()
        .rename(columns={"_week": "week_start"})
    )

    results = []
    for sku, grp in weekly.groupby(group_col):
        grp = grp.sort_values("week_start").reset_index(drop=True)
        slopes = []
        relative_slopes = []
        for i in range(len(grp)):
            start = max(0, i - window_weeks + 1)
            window = grp.iloc[start : i + 1]
            if len(window) >= 3:
                x = np.arange(len(window), dtype=float)
                y = window[value_col].values.astype(float)
                slope, _, _, _, _ = stats.linregress(x, y)
                mean_y = y.mean()
                rel_slope = slope / mean_y if mean_y != 0 else 0.0
            else:
                slope = np.nan
                rel_slope = np.nan
            slopes.append(slope)
            relative_slopes.append(rel_slope)

        grp["trend_slope"] = slopes
        grp["relative_trend_slope"] = relative_slopes
        grp[group_col] = sku
        results.append(grp)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def flag_sparse_demand(
    df: pd.DataFrame,
    value_col: str,
    date_col: str,
    group_col: str,
    zero_threshold: float = 0.6,
) -> pd.Series:
    """
    Flag SKUs with sparse demand (>60% of weeks with zero sales).
    Used by demand_forecast.py to switch to Croston's method.
    Returns Series of bool indexed by group_col.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_week"] = df[date_col].dt.to_period("W").dt.start_time

    weekly = (
        df.groupby([group_col, "_week"])[value_col]
        .sum()
        .reset_index()
    )

    all_weeks = weekly["_week"].unique()
    n_total_weeks = len(all_weeks)

    zero_fracs = {}
    for sku, grp in weekly.groupby(group_col):
        n_zero = (grp[value_col] == 0).sum()
        # Also count weeks with no data at all as zero
        n_missing = n_total_weeks - len(grp)
        total_zero = n_zero + n_missing
        frac_zero = total_zero / n_total_weeks if n_total_weeks > 0 else 0.0
        zero_fracs[sku] = frac_zero

    sparse_flags = pd.Series(zero_fracs, name="is_sparse")
    sparse_flags = sparse_flags >= zero_threshold
    sparse_flags.index.name = group_col
    return sparse_flags
