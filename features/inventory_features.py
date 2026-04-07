"""
Inventory health features.
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_abc_classification(revenue_by_sku: pd.Series) -> pd.Series:
    """
    Compute ABC classification.
    A = top 20% of cumulative revenue
    B = next 30%
    C = remaining 50%
    Returns Series of 'A'/'B'/'C' labels indexed by SKU.
    """
    sorted_rev = revenue_by_sku.sort_values(ascending=False)
    cumulative = sorted_rev.cumsum() / sorted_rev.sum()

    labels = pd.Series(index=sorted_rev.index, dtype=str)
    labels[cumulative <= 0.20] = "A"
    labels[(cumulative > 0.20) & (cumulative <= 0.50)] = "B"
    labels[cumulative > 0.50] = "C"

    # Re-index to match original input order
    return labels.reindex(revenue_by_sku.index)


def compute_xyz_classification(weekly_sales_matrix: pd.DataFrame) -> pd.Series:
    """
    Compute XYZ classification based on coefficient of variation (CV) of weekly sales.
    X = CV < 0.5 (stable, predictable)
    Y = 0.5 <= CV < 1.0 (variable)
    Z = CV >= 1.0 (highly variable, difficult to predict)
    weekly_sales_matrix: DataFrame with SKUs as columns, weeks as index
    Returns Series of 'X'/'Y'/'Z' indexed by SKU.
    """
    means = weekly_sales_matrix.mean(axis=0)
    stds = weekly_sales_matrix.std(axis=0)
    cv = stds / means.replace(0, np.nan)

    labels = pd.Series(index=weekly_sales_matrix.columns, dtype=str)
    labels[cv < 0.5] = "X"
    labels[(cv >= 0.5) & (cv < 1.0)] = "Y"
    labels[cv >= 1.0] = "Z"
    labels[cv.isna()] = "Z"  # zero-demand items are highly unpredictable

    labels.index.name = "sku"
    return labels


def compute_days_cover(stock_on_hand: pd.Series, avg_daily_sales: pd.Series) -> pd.Series:
    """Days of stock remaining at current demand rate. Min 0."""
    days = stock_on_hand / avg_daily_sales.replace(0, np.nan)
    days = days.fillna(0).clip(lower=0)
    days.name = "days_cover"
    return days


def compute_stockout_probability(
    stock_on_hand: pd.Series,
    demand_mean: pd.Series,
    demand_std: pd.Series,
    lead_time_days: pd.Series,
) -> pd.Series:
    """
    Probability that demand during lead time exceeds stock on hand.
    Assumes normal demand distribution during lead time.
    P(stockout) = P(D_LT > SOH) where D_LT ~ N(mean*LT, std*sqrt(LT))
    """
    lt_mean = demand_mean * lead_time_days
    lt_std = demand_std * np.sqrt(lead_time_days)

    probabilities = []
    for soh, mu, sigma in zip(stock_on_hand, lt_mean, lt_std):
        if sigma <= 0 or np.isnan(sigma):
            p = 1.0 if soh < mu else 0.0
        else:
            p = 1.0 - norm.cdf(soh, loc=mu, scale=sigma)
        probabilities.append(p)

    result = pd.Series(probabilities, index=stock_on_hand.index, name="stockout_probability")
    return result.clip(0, 1)


def compute_inventory_features(
    inventory_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all inventory features into a single DataFrame.
    Merges inventory positions with sales-derived demand statistics.
    """
    inv = inventory_df.copy()
    txn = transactions_df.copy()

    if "stock_code" not in inv.columns and "sku_id" in inv.columns:
        inv = inv.rename(columns={"sku_id": "stock_code"})
    if "category_l1" not in txn.columns and "category" in txn.columns:
        txn["category_l1"] = txn["category"]

    sku_col = "stock_code" if "stock_code" in txn.columns else "sku"
    txn["invoice_date"] = pd.to_datetime(txn["invoice_date"])

    # Demand statistics per SKU
    daily_demand = (
        txn[txn["quantity"] > 0]
        .groupby(sku_col)["quantity"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "avg_daily_demand", "std": "demand_std"})
    )
    daily_demand["demand_std"] = daily_demand["demand_std"].fillna(0)

    if sku_col not in inv.columns:
        inv = inv.reset_index()
        if sku_col not in inv.columns and "sku_id" in inv.columns:
            inv = inv.rename(columns={"sku_id": sku_col})

    merged = inv.merge(daily_demand, on=sku_col, how="left", suffixes=("_inventory", "_sales"))

    if "avg_daily_demand_inventory" in merged.columns or "avg_daily_demand_sales" in merged.columns:
        merged["avg_daily_demand"] = merged.get("avg_daily_demand_sales", pd.Series(index=merged.index)).combine_first(
            merged.get("avg_daily_demand_inventory", pd.Series(index=merged.index))
        )
    elif "avg_daily_demand" not in merged.columns:
        merged["avg_daily_demand"] = 0.0

    if "demand_std_sales" in merged.columns or "demand_std_inventory" in merged.columns:
        merged["demand_std"] = merged.get("demand_std_sales", pd.Series(index=merged.index)).combine_first(
            merged.get("demand_std_inventory", pd.Series(index=merged.index))
        )
    elif "demand_std" not in merged.columns:
        merged["demand_std"] = 0.0

    merged["avg_daily_demand"] = merged["avg_daily_demand"].fillna(0)
    merged["demand_std"] = merged["demand_std"].fillna(0)

    soh_col = "stock_on_hand" if "stock_on_hand" in merged.columns else None
    if soh_col:
        merged["days_cover"] = compute_days_cover(
            merged[soh_col], merged["avg_daily_demand"]
        ).values

    if soh_col and "lead_time_days" in merged.columns:
        merged["stockout_probability"] = compute_stockout_probability(
            merged[soh_col],
            merged["avg_daily_demand"],
            merged["demand_std"],
            merged["lead_time_days"],
        ).values

    # Revenue by SKU for ABC
    revenue_by_sku = (
        txn[txn["quantity"] > 0]
        .groupby(sku_col)
        .apply(lambda x: (x["quantity"] * x.get("unit_price_gbp", pd.Series(np.ones(len(x))))).sum())
        .rename("revenue")
    )
    if not revenue_by_sku.empty:
        abc = compute_abc_classification(revenue_by_sku)
        abc_df = abc.reset_index()
        abc_df.columns = [sku_col, "abc_class"]
        merged = merged.merge(abc_df, on=sku_col, how="left", suffixes=("_inventory", "_computed"))
        if "abc_class_computed" in merged.columns or "abc_class_inventory" in merged.columns:
            merged["abc_class"] = merged.get("abc_class_computed", pd.Series(index=merged.index)).combine_first(
                merged.get("abc_class_inventory", pd.Series(index=merged.index))
            )
        merged["abc_class"] = merged["abc_class"].fillna("C")

    # Weekly sales matrix for XYZ
    txn["_week"] = txn["invoice_date"].dt.to_period("W").dt.start_time
    weekly_matrix = (
        txn[txn["quantity"] > 0]
        .groupby([sku_col, "_week"])["quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    if not weekly_matrix.empty:
        xyz = compute_xyz_classification(weekly_matrix.T)
        xyz_df = xyz.reset_index()
        xyz_df.columns = [sku_col, "xyz_class"]
        merged = merged.merge(xyz_df, on=sku_col, how="left", suffixes=("_inventory", "_computed"))
        if "xyz_class_computed" in merged.columns or "xyz_class_inventory" in merged.columns:
            merged["xyz_class"] = merged.get("xyz_class_computed", pd.Series(index=merged.index)).combine_first(
                merged.get("xyz_class_inventory", pd.Series(index=merged.index))
            )
        merged["xyz_class"] = merged["xyz_class"].fillna("Z")

    return merged
