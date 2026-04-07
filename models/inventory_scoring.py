"""Inventory scoring model for stockout and dead-stock risk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from features.inventory_features import compute_inventory_features
from models.model_utils import MODELS_DIR, save_json, save_pickle, standardize_costs, standardize_inventory, standardize_transactions


@dataclass
class InventoryScoringSummary:
    model_name: str
    trained_at: str
    sku_count: int
    reorder_recommended_count: int
    mean_stockout_risk_score: float


class InventoryScoringModel:
    """Score inventory positions using inventory features and anomaly detection."""

    def __init__(self, artifact_dir: Path | None = None) -> None:
        self.artifact_dir = artifact_dir or MODELS_DIR
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_path = self.artifact_dir / "inventory_scoring.pkl"
        self.scores_path = self.artifact_dir / "inventory_scores.parquet"
        self.summary_path = self.artifact_dir / "inventory_scoring_summary.json"

    def train(
        self,
        inventory_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        costs_df: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        inventory = standardize_inventory(inventory_df)
        transactions = standardize_transactions(transactions_df)
        frame = compute_inventory_features(inventory, transactions)
        if costs_df is not None:
            costs = standardize_costs(costs_df)
            merge_columns = [column for column in ["stock_code", "gross_margin_pct", "total_cost_gbp"] if column in costs.columns]
            if "stock_code" in merge_columns:
                frame = frame.merge(costs[merge_columns], on="stock_code", how="left")

        numeric_columns = [
            column
            for column in [
                "stock_on_hand",
                "safety_stock",
                "reorder_point",
                "days_cover",
                "stockout_probability",
                "avg_daily_demand",
                "demand_std",
                "unit_price_gbp",
                "gross_margin_pct",
                "total_cost_gbp",
            ]
            if column in frame.columns
        ]
        model_frame = frame[numeric_columns].fillna(frame[numeric_columns].median())

        scaler = StandardScaler()
        scaled = scaler.fit_transform(model_frame)
        anomaly_model = IsolationForest(random_state=42, contamination=0.15)
        anomaly_model.fit(scaled)
        anomaly_score = -anomaly_model.score_samples(scaled)
        anomaly_scaled = MinMaxScaler().fit_transform(anomaly_score.reshape(-1, 1)).ravel()

        days_cover_score = (1.0 / frame["days_cover"].clip(lower=1.0)).clip(upper=1.0)
        overstock_signal = (frame["days_cover"] > 90).astype(float)
        frame["stockout_risk_score"] = (
            100
            * (
                0.55 * frame.get("stockout_probability", pd.Series(0.0, index=frame.index)).fillna(0.0)
                + 0.25 * days_cover_score.fillna(0.0)
                + 0.20 * pd.Series(anomaly_scaled, index=frame.index)
            )
        ).round(2)
        frame["dead_stock_score"] = (
            100
            * (
                0.60 * overstock_signal
                + 0.25 * pd.Series(anomaly_scaled, index=frame.index)
                + 0.15 * (frame.get("avg_daily_demand", pd.Series(0.0, index=frame.index)) < 1.0).astype(float)
            )
        ).round(2)
        frame["reorder_quantity_suggestion"] = (
            frame.get("reorder_point", pd.Series(0, index=frame.index)).fillna(0)
            - frame.get("stock_on_hand", pd.Series(0, index=frame.index)).fillna(0)
            + frame.get("avg_daily_demand", pd.Series(0.0, index=frame.index)).fillna(0).mul(7).round()
        ).clip(lower=0).astype(int)
        frame["reorder_recommended"] = (
            (frame["stockout_risk_score"] >= 60)
            | (frame["days_cover"].fillna(0) < 14)
            | (frame["reorder_quantity_suggestion"] > 0)
        )

        summary = InventoryScoringSummary(
            model_name="inventory_scoring",
            trained_at=datetime.utcnow().isoformat(),
            sku_count=int(frame["stock_code"].nunique()),
            reorder_recommended_count=int(frame["reorder_recommended"].sum()),
            mean_stockout_risk_score=round(float(frame["stockout_risk_score"].mean()), 2),
        )
        artifact = {
            "scaler": scaler,
            "anomaly_model": anomaly_model,
            "numeric_columns": numeric_columns,
            "summary": summary,
        }
        save_pickle(artifact, self.artifact_path)
        frame.to_parquet(self.scores_path, index=False)
        save_json({"summary": summary}, self.summary_path)
        return {
            "artifact_path": self.artifact_path,
            "scores": frame,
            "summary": summary,
        }