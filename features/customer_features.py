"""
Customer-level RFM and behavioural features.
"""
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def compute_rfm(
    transactions_df: pd.DataFrame,
    reference_date: Optional[date] = None,
    customer_col: str = "customer_id",
    date_col: str = "invoice_date",
    value_col: str = "net_revenue_gbp",
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary value per customer.

    Returns DataFrame with columns:
    - customer_id
    - recency_days: days since last purchase (lower = better)
    - frequency: number of orders in lookback period
    - monetary_value: total spend in lookback period
    - avg_order_value: monetary / frequency
    - r_score, f_score, m_score: quintile scores 1-5
    - rfm_score: composite (r_score * 100 + f_score * 10 + m_score)
    - rfm_segment: string like "555" = champion
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if reference_date is None:
        reference_date = df[date_col].max().date()

    ref_dt = pd.Timestamp(reference_date)
    cutoff = ref_dt - pd.Timedelta(days=lookback_days)

    df_period = df[(df[date_col] >= cutoff) & (df[date_col] <= ref_dt)]

    if value_col not in df_period.columns:
        if "unit_price_gbp" in df_period.columns and "quantity" in df_period.columns:
            df_period = df_period.copy()
            df_period[value_col] = df_period["quantity"] * df_period["unit_price_gbp"]
        else:
            df_period = df_period.copy()
            df_period[value_col] = 1.0

    rfm = (
        df_period[df_period.get("is_return", pd.Series(False, index=df_period.index)) != True]
        .groupby(customer_col)
        .agg(
            last_purchase_date=(date_col, "max"),
            frequency=(date_col, "nunique"),
            monetary_value=(value_col, "sum"),
        )
        .reset_index()
    )

    rfm["recency_days"] = (ref_dt - rfm["last_purchase_date"]).dt.days
    rfm["avg_order_value"] = rfm["monetary_value"] / rfm["frequency"].replace(0, np.nan)
    rfm = rfm.drop(columns=["last_purchase_date"])

    rfm = score_rfm(rfm)

    rfm["rfm_score"] = rfm["r_score"] * 100 + rfm["f_score"] * 10 + rfm["m_score"]
    rfm["rfm_segment"] = (
        rfm["r_score"].astype(str)
        + rfm["f_score"].astype(str)
        + rfm["m_score"].astype(str)
    )

    return rfm


def score_rfm(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quintile scores to RFM DataFrame.
    Note: recency score is INVERTED (lower recency_days = higher score).
    """
    df = rfm_df.copy()

    # Recency: lower is better -> invert labels
    df["r_score"] = pd.qcut(
        df["recency_days"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop"
    ).astype(int)

    df["f_score"] = pd.qcut(
        df["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    ).astype(int)

    df["m_score"] = pd.qcut(
        df["monetary_value"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    ).astype(int)

    return df


def compute_behavioural_features(
    transactions_df: pd.DataFrame,
    reference_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Compute additional behavioural features.

    - avg_basket_size: average number of items per order
    - category_breadth: number of distinct L1 categories purchased
    - channel_preference: most frequent channel
    - days_between_orders: average days between consecutive orders (inter-purchase time)
    - purchase_frequency_trend: is frequency increasing, stable, or declining?
      (compare last 90 days frequency vs 90-180 days ago)
    - preferred_season: season with highest spend
    """
    df = transactions_df.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    if reference_date is None:
        reference_date = df["invoice_date"].max().date()

    ref_dt = pd.Timestamp(reference_date)

    customer_col = "customer_id"
    results = []

    inv_col = "invoice_id" if "invoice_id" in df.columns else "invoice_no" if "invoice_no" in df.columns else None

    for cust, grp in df.groupby(customer_col):
        # Average basket size
        if inv_col:
            basket_sizes = grp.groupby(inv_col)["quantity"].sum()
            avg_basket_size = basket_sizes.mean()
        else:
            avg_basket_size = grp["quantity"].mean()

        # Category breadth
        category_breadth = grp["category_l1"].nunique() if "category_l1" in grp.columns else np.nan

        # Channel preference
        channel_preference = (
            grp["channel"].mode()[0] if "channel" in grp.columns and len(grp) > 0 else "Unknown"
        )

        # Days between orders
        if inv_col:
            order_dates = grp.groupby(inv_col)["invoice_date"].min().sort_values()
        else:
            order_dates = grp["invoice_date"].sort_values()
        diffs = order_dates.diff().dt.days.dropna()
        days_between_orders = diffs.mean() if len(diffs) > 0 else np.nan

        # Purchase frequency trend
        period_recent = grp[grp["invoice_date"] >= ref_dt - pd.Timedelta(days=90)]
        period_prior = grp[
            (grp["invoice_date"] >= ref_dt - pd.Timedelta(days=180))
            & (grp["invoice_date"] < ref_dt - pd.Timedelta(days=90))
        ]
        freq_recent = period_recent["invoice_date"].nunique() if inv_col is None else period_recent.get(inv_col, period_recent["invoice_date"]).nunique()
        freq_prior = period_prior["invoice_date"].nunique() if inv_col is None else period_prior.get(inv_col, period_prior["invoice_date"]).nunique()

        if freq_prior == 0 and freq_recent > 0:
            trend = "increasing"
        elif freq_recent > freq_prior * 1.1:
            trend = "increasing"
        elif freq_recent < freq_prior * 0.9:
            trend = "declining"
        else:
            trend = "stable"

        # Preferred season
        if "net_revenue_gbp" in grp.columns:
            grp = grp.copy()
            grp["_season"] = grp["invoice_date"].dt.month.map(
                lambda m: "Spring" if m in (3, 4, 5)
                else "Summer" if m in (6, 7, 8)
                else "Autumn" if m in (9, 10, 11)
                else "Winter"
            )
            season_spend = grp.groupby("_season")["net_revenue_gbp"].sum()
            preferred_season = season_spend.idxmax() if not season_spend.empty else "Unknown"
        else:
            preferred_season = "Unknown"

        results.append(
            {
                customer_col: cust,
                "avg_basket_size": avg_basket_size,
                "category_breadth": category_breadth,
                "channel_preference": channel_preference,
                "days_between_orders": days_between_orders,
                "purchase_frequency_trend": trend,
                "preferred_season": preferred_season,
            }
        )

    return pd.DataFrame(results)
