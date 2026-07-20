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
    transactions_df,
    reference_date=None,
):
    """Compute customer behavioural features (vectorised).

    - avg_basket_size, category_breadth, channel_preference, days_between_orders,
      purchase_frequency_trend (recent 90d vs prior 90-180d), preferred_season.
    """
    import numpy as np
    import pandas as pd

    df = transactions_df.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    if "category_l1" not in df.columns and "category" in df.columns:
        df["category_l1"] = df["category"]
    if reference_date is None:
        reference_date = df["invoice_date"].max().date()
    ref_dt = pd.Timestamp(reference_date)

    inv_col = "invoice_id" if "invoice_id" in df.columns else "invoice_no" if "invoice_no" in df.columns else None

    # Basket size: units per order, averaged per customer.
    if inv_col:
        basket = df.groupby(["customer_id", inv_col])["quantity"].sum().groupby("customer_id").mean()
    else:
        basket = df.groupby("customer_id")["quantity"].mean()
    basket = basket.rename("avg_basket_size")

    # Category breadth.
    if "category_l1" in df.columns:
        breadth = df.groupby("customer_id")["category_l1"].nunique().rename("category_breadth")
    else:
        breadth = pd.Series(np.nan, index=basket.index, name="category_breadth")

    # Channel preference (mode).
    if "channel" in df.columns:
        channel = df.groupby("customer_id")["channel"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown").rename("channel_preference")
    else:
        channel = pd.Series("Unknown", index=basket.index, name="channel_preference")

    # Days between orders: mean gap between distinct order dates.
    if inv_col:
        order_dates = df.groupby(["customer_id", inv_col])["invoice_date"].min().reset_index()
    else:
        order_dates = df[["customer_id", "invoice_date"]].copy()
    order_dates = order_dates.sort_values(["customer_id", "invoice_date"])
    order_dates["gap"] = order_dates.groupby("customer_id")["invoice_date"].diff().dt.days
    ipt = order_dates.groupby("customer_id")["gap"].mean().rename("days_between_orders")

    # Purchase frequency trend: distinct orders recent (<=90d) vs prior (90-180d).
    def _orders_in(window_df):
        if inv_col:
            return window_df.groupby("customer_id")[inv_col].nunique()
        return window_df.groupby("customer_id")["invoice_date"].nunique()

    recent = df[df["invoice_date"] >= ref_dt - pd.Timedelta(days=90)]
    prior = df[(df["invoice_date"] >= ref_dt - pd.Timedelta(days=180)) & (df["invoice_date"] < ref_dt - pd.Timedelta(days=90))]
    freq_recent = _orders_in(recent).rename("fr")
    freq_prior = _orders_in(prior).rename("fp")
    trend_df = pd.concat([freq_recent, freq_prior], axis=1).reindex(basket.index).fillna(0)
    conditions = [
        (trend_df["fp"] == 0) & (trend_df["fr"] > 0),
        trend_df["fr"] > trend_df["fp"] * 1.1,
        trend_df["fr"] < trend_df["fp"] * 0.9,
    ]
    trend = pd.Series(np.select(conditions, ["increasing", "increasing", "declining"], default="stable"), index=trend_df.index, name="purchase_frequency_trend")

    # Preferred season by spend.
    if "net_revenue_gbp" in df.columns:
        season = df["invoice_date"].dt.month.map(
            lambda m: "Spring" if m in (3, 4, 5) else "Summer" if m in (6, 7, 8) else "Autumn" if m in (9, 10, 11) else "Winter"
        )
        season_spend = df.assign(_season=season).groupby(["customer_id", "_season"])["net_revenue_gbp"].sum()
        preferred = season_spend.groupby("customer_id").idxmax().map(lambda t: t[1]).rename("preferred_season")
    else:
        preferred = pd.Series("Unknown", index=basket.index, name="preferred_season")

    out = pd.concat([basket, breadth, channel, ipt, trend, preferred], axis=1).reset_index()
    return out
