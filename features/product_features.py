"""
Product-level features for demand forecasting and ranking.
"""
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def compute_product_features(
    transactions_df: pd.DataFrame,
    inventory_df: Optional[pd.DataFrame] = None,
    costs_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute per-SKU product features.

    Returns DataFrame indexed by stock_code/sku with features:
    - total_units_sold_12m, total_revenue_12m
    - avg_unit_price, price_std, price_cv (coefficient of variation of price)
    - n_orders_12m, n_distinct_customers_12m
    - return_rate (returns / total orders)
    - avg_basket_position (what position in basket this product tends to appear)
    - is_seasonal (CV of monthly sales > 0.3)
    - peak_month (month with highest avg sales)
    - category_l1, brand (from product metadata)
    - unit_margin_pct (from costs_df if available)
    """
    df = transactions_df.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    if "category_l1" not in df.columns and "category" in df.columns:
        df["category_l1"] = df["category"]

    cutoff = df["invoice_date"].max() - pd.DateOffset(months=12)
    df12 = df[df["invoice_date"] >= cutoff]

    sku_col = "stock_code" if "stock_code" in df.columns else "sku"

    agg = (
        df12.groupby(sku_col)
        .agg(
            total_units_sold_12m=("quantity", lambda x: x[x > 0].sum()),
            total_revenue_12m=(
                "net_revenue_gbp",
                lambda x: x[x > 0].sum() if "net_revenue_gbp" in df12.columns else 0,
            ),
            avg_unit_price=("unit_price_gbp", "mean"),
            price_std=("unit_price_gbp", "std"),
            n_orders_12m=("invoice_date", "nunique"),
        )
        .reset_index()
    )

    if "net_revenue_gbp" not in df12.columns and "unit_price_gbp" in df12.columns:
        agg["total_revenue_12m"] = (
            df12[df12["quantity"] > 0]
            .groupby(sku_col)
            .apply(lambda x: (x["quantity"] * x["unit_price_gbp"]).sum())
            .reset_index(drop=True)
        )

    agg["price_cv"] = agg["price_std"] / agg["avg_unit_price"].replace(0, np.nan)

    if "customer_id" in df12.columns:
        cust_counts = (
            df12.groupby(sku_col)["customer_id"].nunique().reset_index()
        )
        cust_counts.columns = [sku_col, "n_distinct_customers_12m"]
        agg = agg.merge(cust_counts, on=sku_col, how="left")
    else:
        agg["n_distinct_customers_12m"] = np.nan

    if "is_return" in df12.columns:
        return_rates = (
            df12.groupby(sku_col)["is_return"]
            .mean()
            .reset_index()
            .rename(columns={"is_return": "return_rate"})
        )
        agg = agg.merge(return_rates, on=sku_col, how="left")
    else:
        agg["return_rate"] = 0.0

    # Basket position: average rank of product within each invoice
    if "invoice_id" in df12.columns or "invoice_no" in df12.columns:
        inv_col = "invoice_id" if "invoice_id" in df12.columns else "invoice_no"
        df12 = df12.copy()
        df12["basket_position"] = df12.groupby(inv_col).cumcount() + 1
        basket_pos = (
            df12.groupby(sku_col)["basket_position"]
            .mean()
            .reset_index()
            .rename(columns={"basket_position": "avg_basket_position"})
        )
        agg = agg.merge(basket_pos, on=sku_col, how="left")
    else:
        agg["avg_basket_position"] = np.nan

    # Monthly sales CV for seasonality
    df12 = df12.copy()
    df12["_month"] = df12["invoice_date"].dt.month
    monthly_sales = (
        df12[df12["quantity"] > 0]
        .groupby([sku_col, "_month"])["quantity"]
        .sum()
        .reset_index()
    )
    monthly_cv = (
        monthly_sales.groupby(sku_col)["quantity"]
        .agg(["mean", "std"])
        .reset_index()
    )
    monthly_cv["is_seasonal"] = (monthly_cv["std"] / monthly_cv["mean"].replace(0, np.nan)) > 0.3
    monthly_cv = monthly_cv[[sku_col, "is_seasonal"]]
    agg = agg.merge(monthly_cv, on=sku_col, how="left")
    agg["is_seasonal"] = agg["is_seasonal"].fillna(False)

    peak_month = (
        monthly_sales.loc[
            monthly_sales.groupby(sku_col)["quantity"].idxmax(), [sku_col, "_month"]
        ]
        .rename(columns={"_month": "peak_month"})
    )
    agg = agg.merge(peak_month, on=sku_col, how="left")

    if "category_l1" in df12.columns:
        cat = (
            df12.groupby(sku_col)["category_l1"]
            .agg(lambda x: x.mode()[0] if len(x) > 0 else "Unknown")
            .reset_index()
        )
        agg = agg.merge(cat, on=sku_col, how="left")
    else:
        agg["category_l1"] = "Unknown"

    if "brand" in df12.columns:
        brand = (
            df12.groupby(sku_col)["brand"]
            .agg(lambda x: x.mode()[0] if len(x) > 0 else "Unknown")
            .reset_index()
        )
        agg = agg.merge(brand, on=sku_col, how="left")
    else:
        agg["brand"] = "Unknown"

    if costs_df is not None:
        costs_df = costs_df.copy()
        if sku_col not in costs_df.columns and "sku_id" in costs_df.columns:
            costs_df = costs_df.rename(columns={"sku_id": sku_col})

    if costs_df is not None and sku_col in costs_df.columns and "unit_cost_gbp" in costs_df.columns:
        costs_df = costs_df[[sku_col, "unit_cost_gbp"]].copy()
        agg = agg.merge(costs_df, on=sku_col, how="left")
        agg["unit_margin_pct"] = (
            (agg["avg_unit_price"] - agg["unit_cost_gbp"]) / agg["avg_unit_price"].replace(0, np.nan)
        ) * 100
        agg = agg.drop(columns=["unit_cost_gbp"])
    else:
        agg["unit_margin_pct"] = np.nan

    agg = agg.set_index(sku_col)
    return agg


def compute_price_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price-related features per SKU.

    - price_elasticity_estimate: correlation of price changes with quantity changes (OLS)
    - avg_discount_pct: mean discount when promotional
    - promo_frequency: % of transactions that are promotional
    - price_band: Budget (<£5), Mid (£5-£20), Premium (£20-£50), Luxury (>£50)
    """
    df = transactions_df.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    sku_col = "stock_code" if "stock_code" in df.columns else "sku"

    results = []

    for sku, grp in df.groupby(sku_col):
        grp = grp.sort_values("invoice_date")
        avg_price = grp["unit_price_gbp"].mean()

        # Price band
        if avg_price < 5:
            price_band = "Budget"
        elif avg_price < 20:
            price_band = "Mid"
        elif avg_price < 50:
            price_band = "Premium"
        else:
            price_band = "Luxury"

        # Price elasticity: OLS of pct price change vs pct quantity change
        price_changes = grp["unit_price_gbp"].pct_change().dropna()
        qty_changes = grp["quantity"].pct_change().dropna()
        if len(price_changes) > 2 and price_changes.std() > 0:
            valid = (~price_changes.isin([np.inf, -np.inf])) & (~qty_changes.isin([np.inf, -np.inf]))
            price_changes = price_changes[valid]
            qty_changes = qty_changes[valid]
            if len(price_changes) > 1:
                from numpy.polynomial import polynomial as P

                coeffs = np.polyfit(price_changes.values, qty_changes.values, 1)
                elasticity = coeffs[0]
            else:
                elasticity = np.nan
        else:
            elasticity = np.nan

        # Promo detection: price > 10% below median
        median_price = grp["unit_price_gbp"].median()
        promo_mask = grp["unit_price_gbp"] < median_price * 0.9
        promo_frequency = promo_mask.mean()

        if "discount_pct" in grp.columns:
            avg_discount_pct = grp.loc[promo_mask, "discount_pct"].mean() if promo_mask.any() else 0.0
        else:
            avg_discount_pct = (
                ((median_price - grp.loc[promo_mask, "unit_price_gbp"]) / median_price * 100).mean()
                if promo_mask.any()
                else 0.0
            )

        results.append(
            {
                sku_col: sku,
                "avg_unit_price": avg_price,
                "price_band": price_band,
                "price_elasticity_estimate": elasticity,
                "avg_discount_pct": avg_discount_pct,
                "promo_frequency": promo_frequency,
            }
        )

    out = pd.DataFrame(results).set_index(sku_col)
    return out
