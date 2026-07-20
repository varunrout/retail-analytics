"""Trend detection for emerging and declining retail demand."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk

from features.demand_features import compute_rolling_demand, compute_trend_slope
from models.model_utils import MODELS_DIR, save_json, standardize_transactions


@dataclass
class TrendDetectionSummary:
    model_name: str
    trained_at: str
    sku_count: int
    accelerating_count: int
    declining_count: int


class TrendDetectionModel:
    """Detect growth, decline, and spike signals across SKUs and categories.

    Trend significance is tested with the Mann-Kendall non-parametric monotonic
    trend test on each SKU\'s weekly series (p < 0.05), combined with a recent
    z-score. A SKU is only labelled accelerating/declining when the Mann-Kendall
    test is significant, so labels reflect a statistical signal, not a raw slope.
    """

    MK_WINDOW_WEEKS = 26
    MK_ALPHA = 0.05

    def __init__(self, artifact_dir: Path | None = None) -> None:
        self.artifact_dir = artifact_dir or MODELS_DIR
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.sku_path = self.artifact_dir / "trend_detection_sku.parquet"
        self.category_path = self.artifact_dir / "trend_detection_category.parquet"
        self.summary_path = self.artifact_dir / "trend_detection_summary.json"


    def _mann_kendall(self, weekly: pd.DataFrame) -> pd.DataFrame:
        """Per-SKU Mann-Kendall test on the recent weekly series."""
        rows: list[dict[str, object]] = []
        for stock_code, group in weekly.sort_values("week_start").groupby("stock_code"):
            series = group["quantity"].tail(self.MK_WINDOW_WEEKS).to_numpy(dtype=float)
            if len(series) < 8 or np.all(series == series[0]):
                rows.append({"stock_code": stock_code, "mk_trend": "no trend", "mk_pvalue": 1.0, "mk_slope": 0.0})
                continue
            try:
                result = mk.original_test(series, alpha=self.MK_ALPHA)
                rows.append(
                    {
                        "stock_code": stock_code,
                        "mk_trend": result.trend,
                        "mk_pvalue": float(result.p),
                        "mk_slope": float(result.slope),
                    }
                )
            except Exception:
                rows.append({"stock_code": stock_code, "mk_trend": "no trend", "mk_pvalue": 1.0, "mk_slope": 0.0})
        return pd.DataFrame(rows)

    def run(self, transactions_df: pd.DataFrame) -> dict[str, object]:
        transactions = standardize_transactions(transactions_df)
        rolling = compute_rolling_demand(transactions, value_col="quantity", date_col="invoice_date", group_col="stock_code")
        slopes = compute_trend_slope(transactions, value_col="quantity", date_col="invoice_date", group_col="stock_code")
        weekly = (
            transactions[transactions["quantity"] > 0]
            .assign(week_start=lambda data: data["invoice_date"].dt.to_period("W").dt.start_time)
            .groupby(["stock_code", "week_start"], as_index=False)
            .agg(quantity=("quantity", "sum"), category_l1=("category_l1", lambda values: values.mode().iat[0]))
        )

        recent_sales = (
            weekly.sort_values("week_start")
            .groupby("stock_code")
            .tail(4)
            .groupby("stock_code", as_index=False)["quantity"]
            .mean()
            .rename(columns={"quantity": "recent_4w_avg"})
        )
        baseline = (
            weekly.groupby("stock_code", as_index=False)["quantity"]
            .agg(baseline_mean="mean", baseline_std="std")
            .fillna(0.0)
        )
        latest_slopes = slopes.sort_values("week_start").groupby("stock_code").tail(1)
        latest_rolling = rolling.sort_values("week_start").groupby("stock_code").tail(1)

        sku_trends = (
            latest_slopes.merge(latest_rolling[["stock_code", "rolling_28d_units", "rolling_28d_cv"]], on="stock_code", how="left")
            .merge(recent_sales, on="stock_code", how="left")
            .merge(baseline, on="stock_code", how="left")
            .merge(weekly[["stock_code", "category_l1"]].drop_duplicates(), on="stock_code", how="left")
        )
        sku_trends["z_score_recent_sales"] = (
            (sku_trends["recent_4w_avg"] - sku_trends["baseline_mean"])
            / sku_trends["baseline_std"].replace(0.0, np.nan)
        ).fillna(0.0)
        mk_stats = self._mann_kendall(weekly)
        sku_trends = sku_trends.merge(mk_stats, on="stock_code", how="left")
        sku_trends["mk_trend"] = sku_trends["mk_trend"].fillna("no trend")
        sku_trends["mk_pvalue"] = sku_trends["mk_pvalue"].fillna(1.0)
        sku_trends["mk_significant"] = sku_trends["mk_pvalue"] < self.MK_ALPHA
        sku_trends["trend_label"] = np.select(
            [
                (sku_trends["mk_trend"] == "increasing")
                & sku_trends["mk_significant"]
                & (sku_trends["z_score_recent_sales"] >= 1.5),
                (sku_trends["mk_trend"] == "increasing") & sku_trends["mk_significant"],
                (sku_trends["mk_trend"] == "decreasing") & sku_trends["mk_significant"],
            ],
            ["accelerating", "growing", "declining"],
            default="stable",
        )
        # Ranked "act on this" priority: significant, fast-moving SKUs first.
        sku_trends["priority_score"] = (
            sku_trends["mk_slope"].abs().fillna(0.0)
            * (-np.log10(sku_trends["mk_pvalue"].clip(lower=1e-6)))
        )
        sku_trends["action_rank"] = (
            sku_trends["priority_score"].rank(ascending=False, method="first").astype(int)
        )
        category_trends = (
            sku_trends.groupby("category_l1", as_index=False)
            .agg(
                sku_count=("stock_code", "nunique"),
                avg_relative_trend_slope=("relative_trend_slope", "mean"),
                avg_recent_sales_z=("z_score_recent_sales", "mean"),
            )
            .sort_values("avg_relative_trend_slope", ascending=False)
        )
        category_trends["category_label"] = np.select(
            [
                category_trends["avg_relative_trend_slope"] >= 0.02,
                category_trends["avg_relative_trend_slope"] <= -0.01,
            ],
            ["hot", "cooling"],
            default="steady",
        )

        summary = TrendDetectionSummary(
            model_name="trend_detection",
            trained_at=datetime.utcnow().isoformat(),
            sku_count=int(sku_trends["stock_code"].nunique()),
            accelerating_count=int((sku_trends["trend_label"] == "accelerating").sum()),
            declining_count=int((sku_trends["trend_label"] == "declining").sum()),
        )
        sku_trends.to_parquet(self.sku_path, index=False)
        category_trends.to_parquet(self.category_path, index=False)
        save_json({"summary": summary}, self.summary_path)
        return {
            "sku_trends": sku_trends,
            "category_trends": category_trends,
            "summary": summary,
        }