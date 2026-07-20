"""Demand forecasting model for SKU-level weekly demand.

The forecaster is deliberately simple: a recursive per-SKU Ridge regression on
lag and calendar features, with a seasonal-naive fallback for short histories.
Its value is not absolute accuracy (weekly SKU demand is intermittent, so MAPE
is high) but *lift over a naive baseline*: every trained SKU is scored against a
seasonal-naive forecast on the identical held-out weeks, and we report the share
of SKUs the model actually beats. Each forecast ends in a reorder-point decision,
not a bare number.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from models.model_utils import MODELS_DIR, compute_mape, save_json, save_pickle, standardize_transactions

# Weeks of history held out for validation, and default replenishment lead time.
VALIDATION_WEEKS = 4
DEFAULT_LEAD_TIME_WEEKS = 2
# z-score for the safety-stock service level (~97.5%).
SERVICE_LEVEL_Z = 1.96
FEATURE_COLUMNS = [
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


@dataclass
class DemandForecastSummary:
    model_name: str
    trained_at: str
    sku_count: int
    forecast_horizon_weeks: int
    mean_validation_mape: float
    baseline_mean_mape: float
    model_wmape: float
    baseline_wmape: float
    share_beating_baseline: float
    fallback_sku_count: int


def _wmape(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.abs(actual).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(actual - forecast).sum() / denom)


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

    def _seasonal_naive(self, sku_history: pd.DataFrame, holdout_dates: pd.Series) -> np.ndarray:
        """Seasonal-naive baseline: units sold 52 weeks before each holdout week.

        Falls back to the mean of the last four training weeks when there is not a
        full year of prior history for a given holdout week.
        """
        history = sku_history.set_index("week_start")["units_sold"]
        train = history[history.index < holdout_dates.min()]
        fallback = float(train.tail(4).mean()) if not train.empty else 0.0
        preds: list[float] = []
        for date in holdout_dates:
            lag_date = date - pd.Timedelta(weeks=52)
            if lag_date in history.index:
                preds.append(float(history.loc[lag_date]))
            else:
                preds.append(fallback)
        return np.asarray(preds, dtype=float)

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
    ) -> tuple[list[dict[str, object]], float]:
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
        return forecasts, uncertainty

    def _reorder_point(
        self,
        forecasts: list[dict[str, object]],
        uncertainty: float,
        lead_time_weeks: int,
    ) -> float:
        """Translate the forecast into a decision: the reorder point.

        reorder_point = expected demand over the lead time + safety stock, where
        safety stock covers forecast uncertainty at the target service level.
        """
        weeks = min(lead_time_weeks, len(forecasts))
        lead_time_demand = sum(float(row["forecast_units"]) for row in forecasts[:weeks])
        safety_stock = SERVICE_LEVEL_Z * uncertainty * np.sqrt(max(weeks, 1))
        return float(np.ceil(lead_time_demand + safety_stock))

    def train(
        self,
        transactions_df: pd.DataFrame,
        forecast_horizon_weeks: int = 8,
        min_history_weeks: int = 16,
        lead_time_weeks: int = DEFAULT_LEAD_TIME_WEEKS,
    ) -> dict[str, object]:
        history = self._build_weekly_history(transactions_df)
        if history.empty:
            raise ValueError("No transaction history available for demand forecasting.")

        feature_frame = self._feature_frame(history)
        feature_columns = FEATURE_COLUMNS
        training = feature_frame.dropna(subset=feature_columns + ["units_sold"]).copy()

        models: dict[str, Ridge] = {}
        fallback_skus: list[str] = []
        metric_rows: list[dict[str, object]] = []
        forecast_rows: list[dict[str, object]] = []
        pooled_actual: list[float] = []
        pooled_model: list[float] = []
        pooled_baseline: list[float] = []

        for stock_code, sku_frame in training.groupby("stock_code"):
            sku_frame = sku_frame.sort_values("week_start")
            sku_history = history[history["stock_code"] == stock_code].sort_values("week_start")
            model: Ridge | None = None
            method_used = "seasonal_naive"
            validation_mape = 0.0
            baseline_mape = 0.0
            beats_baseline = False

            if len(sku_frame) >= min_history_weeks:
                split_index = max(len(sku_frame) - VALIDATION_WEEKS, 1)
                train_frame = sku_frame.iloc[:split_index]
                valid_frame = sku_frame.iloc[split_index:]
                if len(train_frame) >= 8 and len(valid_frame) > 0:
                    model = Ridge(alpha=1.0)
                    model.fit(train_frame[feature_columns], train_frame["units_sold"])
                    validation_preds = pd.Series(
                        model.predict(valid_frame[feature_columns]), index=valid_frame.index
                    ).clip(lower=0.0)
                    validation_mape = compute_mape(valid_frame["units_sold"], validation_preds)

                    # Seasonal-naive baseline on the IDENTICAL held-out weeks.
                    baseline_preds = self._seasonal_naive(sku_history, valid_frame["week_start"])
                    baseline_mape = compute_mape(valid_frame["units_sold"], pd.Series(baseline_preds))
                    beats_baseline = bool(validation_mape < baseline_mape)

                    actual = valid_frame["units_sold"].to_numpy(dtype=float)
                    pooled_actual.extend(actual.tolist())
                    pooled_model.extend(validation_preds.to_numpy(dtype=float).tolist())
                    pooled_baseline.extend(baseline_preds.tolist())

                    model.fit(sku_frame[feature_columns], sku_frame["units_sold"])
                    models[stock_code] = model
                    method_used = "ridge_recursive"
                else:
                    fallback_skus.append(stock_code)
            else:
                fallback_skus.append(stock_code)

            sku_forecasts, uncertainty = self._forecast_single_sku(
                stock_code=stock_code,
                history=sku_history,
                forecast_horizon_weeks=forecast_horizon_weeks,
                model=model,
                feature_columns=feature_columns,
                method_used=method_used,
            )
            reorder_point = self._reorder_point(sku_forecasts, uncertainty, lead_time_weeks)
            for row in sku_forecasts:
                row["reorder_point"] = reorder_point
            forecast_rows.extend(sku_forecasts)

            metric_rows.append(
                {
                    "stock_code": stock_code,
                    "history_weeks": int(len(sku_history)),
                    "method_used": method_used,
                    "validation_mape": round(float(validation_mape), 4),
                    "baseline_mape": round(float(baseline_mape), 4),
                    "beats_baseline": beats_baseline,
                    "recent_weekly_units": round(float(sku_history["units_sold"].tail(4).mean()), 2),
                    "reorder_point": reorder_point,
                }
            )

        metrics_df = pd.DataFrame(metric_rows).sort_values("stock_code")
        forecasts_df = pd.DataFrame(forecast_rows).sort_values(["stock_code", "forecast_date"])

        evaluated = metrics_df[metrics_df["method_used"] == "ridge_recursive"]
        model_wmape = _wmape(np.array(pooled_actual), np.array(pooled_model)) if pooled_actual else float("nan")
        baseline_wmape = _wmape(np.array(pooled_actual), np.array(pooled_baseline)) if pooled_actual else float("nan")
        share_beating = float(evaluated["beats_baseline"].mean()) if not evaluated.empty else 0.0

        summary = DemandForecastSummary(
            model_name="demand_forecast",
            trained_at=datetime.utcnow().isoformat(),
            sku_count=int(metrics_df["stock_code"].nunique()),
            forecast_horizon_weeks=forecast_horizon_weeks,
            mean_validation_mape=round(float(evaluated["validation_mape"].mean()), 4) if not evaluated.empty else 0.0,
            baseline_mean_mape=round(float(evaluated["baseline_mape"].mean()), 4) if not evaluated.empty else 0.0,
            model_wmape=round(model_wmape, 4),
            baseline_wmape=round(baseline_wmape, 4),
            share_beating_baseline=round(share_beating, 4),
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
