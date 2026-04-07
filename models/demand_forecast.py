"""Demand forecasting model for SKU-level weekly demand."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from models.model_utils import MODELS_DIR, compute_mape, save_json, save_pickle, standardize_transactions


@dataclass
class DemandForecastSummary:
    model_name: str
    trained_at: str
    sku_count: int
    forecast_horizon_weeks: int
    mean_validation_mape: float
    fallback_sku_count: int


class DemandForecastModel:
    """Train a lightweight recursive weekly demand forecast per SKU."""

    def __init__(self, artifact_dir: Path | None = None) -> None:
        self.artifact_dir = artifact_dir or MODELS_DIR
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_path = self.artifact_dir / "demand_forecast.pkl"
        self.metrics_path = self.artifact_dir / "demand_forecast_metrics.parquet"
        self.forecasts_path = self.artifact_dir / "demand_forecast_predictions.parquet"
        self.summary_path = self.artifact_dir / "demand_forecast_summary.json"

    def _build_weekly_history(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        transactions = standardize_transactions(transactions_df)
        positive_sales = transactions[transactions["quantity"] > 0].copy()
        positive_sales["week_start"] = positive_sales["invoice_date"].dt.to_period("W").dt.start_time

        aggregated = (
            positive_sales.groupby(["stock_code", "week_start"], as_index=False)
            .agg(
                units_sold=("quantity", "sum"),
                avg_unit_price=("unit_price_gbp", "mean"),
                category_l1=("category_l1", lambda values: values.mode().iat[0]),
                brand=("brand", lambda values: values.mode().iat[0]),
            )
            .sort_values(["stock_code", "week_start"])
        )

        completed: list[pd.DataFrame] = []
        for stock_code, group in aggregated.groupby("stock_code"):
            weekly_index = pd.date_range(group["week_start"].min(), group["week_start"].max(), freq="W-MON")
            weekly = group.set_index("week_start").reindex(weekly_index)
            weekly.index.name = "week_start"
            weekly["stock_code"] = stock_code
            weekly["units_sold"] = weekly["units_sold"].fillna(0.0)
            weekly["avg_unit_price"] = weekly["avg_unit_price"].ffill().bfill().fillna(0.0)
            weekly["category_l1"] = group["category_l1"].mode().iat[0]
            weekly["brand"] = group["brand"].mode().iat[0]
            completed.append(weekly.reset_index())
        return pd.concat(completed, ignore_index=True) if completed else pd.DataFrame()

    def _feature_frame(self, history_df: pd.DataFrame) -> pd.DataFrame:
        feature_frames: list[pd.DataFrame] = []
        for stock_code, group in history_df.groupby("stock_code"):
            frame = group.sort_values("week_start").copy()
            frame["lag_1"] = frame["units_sold"].shift(1)
            frame["lag_2"] = frame["units_sold"].shift(2)
            frame["lag_4"] = frame["units_sold"].shift(4)
            frame["lag_8"] = frame["units_sold"].shift(8)
            frame["rolling_mean_4"] = frame["units_sold"].shift(1).rolling(4, min_periods=1).mean()
            frame["rolling_std_4"] = frame["units_sold"].shift(1).rolling(4, min_periods=1).std().fillna(0.0)
            frame["rolling_mean_12"] = frame["units_sold"].shift(1).rolling(12, min_periods=1).mean()
            frame["month"] = frame["week_start"].dt.month
            frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
            frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)
            frame["stock_code"] = stock_code
            feature_frames.append(frame)
        return pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()

    def _fallback_value(self, history: pd.DataFrame, forecast_date: pd.Timestamp) -> float:
        recent = history.tail(8)
        same_month = history[history["week_start"].dt.month == forecast_date.month]
        base_value = recent["units_sold"].mean() if not recent.empty else history["units_sold"].mean()
        if not same_month.empty and history["units_sold"].mean() > 0:
            seasonality = same_month["units_sold"].mean() / history["units_sold"].mean()
            base_value = base_value * seasonality
        return max(float(base_value), 0.0)

    def _forecast_single_sku(
        self,
        stock_code: str,
        history: pd.DataFrame,
        forecast_horizon_weeks: int,
        model: Ridge | None,
        feature_columns: list[str],
        method_used: str,
    ) -> list[dict[str, object]]:
        working = history.sort_values("week_start").copy()
        forecasts: list[dict[str, object]] = []
        uncertainty = max(float(working["units_sold"].tail(12).std(ddof=0)), 1.0)

        for _ in range(forecast_horizon_weeks):
            next_date = working["week_start"].max() + pd.Timedelta(weeks=1)
            feature_row = {
                "lag_1": float(working["units_sold"].iloc[-1]),
                "lag_2": float(working["units_sold"].iloc[-2]) if len(working) >= 2 else float(working["units_sold"].iloc[-1]),
                "lag_4": float(working["units_sold"].iloc[-4]) if len(working) >= 4 else float(working["units_sold"].tail(4).mean()),
                "lag_8": float(working["units_sold"].iloc[-8]) if len(working) >= 8 else float(working["units_sold"].mean()),
                "rolling_mean_4": float(working["units_sold"].tail(4).mean()),
                "rolling_std_4": float(working["units_sold"].tail(4).std(ddof=0)),
                "rolling_mean_12": float(working["units_sold"].tail(12).mean()),
                "month_sin": float(np.sin(2 * np.pi * next_date.month / 12)),
                "month_cos": float(np.cos(2 * np.pi * next_date.month / 12)),
            }
            if model is None:
                prediction = self._fallback_value(working, next_date)
            else:
                prediction = float(model.predict(pd.DataFrame([feature_row], columns=feature_columns))[0])
            prediction = max(prediction, 0.0)
            forecasts.append(
                {
                    "stock_code": stock_code,
                    "forecast_date": next_date,
                    "forecast_units": round(prediction, 2),
                    "forecast_lower": round(max(prediction - uncertainty, 0.0), 2),
                    "forecast_upper": round(prediction + uncertainty, 2),
                    "method_used": method_used,
                }
            )
            working = pd.concat(
                [
                    working,
                    pd.DataFrame(
                        [
                            {
                                "week_start": next_date,
                                "stock_code": stock_code,
                                "units_sold": prediction,
                                "avg_unit_price": float(working["avg_unit_price"].iloc[-1]),
                                "category_l1": working["category_l1"].iloc[-1],
                                "brand": working["brand"].iloc[-1],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        return forecasts

    def train(
        self,
        transactions_df: pd.DataFrame,
        forecast_horizon_weeks: int = 8,
        min_history_weeks: int = 16,
    ) -> dict[str, object]:
        history = self._build_weekly_history(transactions_df)
        if history.empty:
            raise ValueError("No transaction history available for demand forecasting.")

        feature_frame = self._feature_frame(history)
        feature_columns = [
            "lag_1",
            "lag_2",
            "lag_4",
            "lag_8",
            "rolling_mean_4",
            "rolling_std_4",
            "rolling_mean_12",
            "month_sin",
            "month_cos",
        ]
        training = feature_frame.dropna(subset=feature_columns + ["units_sold"]).copy()

        models: dict[str, Ridge] = {}
        fallback_skus: list[str] = []
        metric_rows: list[dict[str, object]] = []
        forecast_rows: list[dict[str, object]] = []

        for stock_code, sku_frame in training.groupby("stock_code"):
            sku_frame = sku_frame.sort_values("week_start")
            sku_history = history[history["stock_code"] == stock_code].sort_values("week_start")
            model: Ridge | None = None
            method_used = "seasonal_naive"
            validation_mape = 0.0

            if len(sku_frame) >= min_history_weeks:
                split_index = max(len(sku_frame) - 4, 1)
                train_frame = sku_frame.iloc[:split_index]
                valid_frame = sku_frame.iloc[split_index:]
                if len(train_frame) >= 8 and len(valid_frame) > 0:
                    model = Ridge(alpha=1.0)
                    model.fit(train_frame[feature_columns], train_frame["units_sold"])
                    validation_preds = pd.Series(model.predict(valid_frame[feature_columns]), index=valid_frame.index).clip(lower=0.0)
                    validation_mape = compute_mape(valid_frame["units_sold"], validation_preds)
                    model.fit(sku_frame[feature_columns], sku_frame["units_sold"])
                    models[stock_code] = model
                    method_used = "ridge_recursive"
                else:
                    fallback_skus.append(stock_code)
            else:
                fallback_skus.append(stock_code)

            metric_rows.append(
                {
                    "stock_code": stock_code,
                    "history_weeks": int(len(sku_history)),
                    "method_used": method_used,
                    "validation_mape": round(float(validation_mape), 4),
                    "recent_weekly_units": round(float(sku_history["units_sold"].tail(4).mean()), 2),
                }
            )
            forecast_rows.extend(
                self._forecast_single_sku(
                    stock_code=stock_code,
                    history=sku_history,
                    forecast_horizon_weeks=forecast_horizon_weeks,
                    model=model,
                    feature_columns=feature_columns,
                    method_used=method_used,
                )
            )

        metrics_df = pd.DataFrame(metric_rows).sort_values("stock_code")
        forecasts_df = pd.DataFrame(forecast_rows).sort_values(["stock_code", "forecast_date"])
        summary = DemandForecastSummary(
            model_name="demand_forecast",
            trained_at=datetime.utcnow().isoformat(),
            sku_count=int(metrics_df["stock_code"].nunique()),
            forecast_horizon_weeks=forecast_horizon_weeks,
            mean_validation_mape=round(float(metrics_df["validation_mape"].mean()), 4),
            fallback_sku_count=len(fallback_skus),
        )

        artifact = {
            "feature_columns": feature_columns,
            "models": models,
            "summary": summary,
        }
        save_pickle(artifact, self.artifact_path)
        metrics_df.to_parquet(self.metrics_path, index=False)
        forecasts_df.to_parquet(self.forecasts_path, index=False)
        save_json({"summary": summary}, self.summary_path)

        return {
            "artifact_path": self.artifact_path,
            "metrics": metrics_df,
            "forecasts": forecasts_df,
            "summary": summary,
        }