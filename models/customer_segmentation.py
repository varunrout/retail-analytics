"""Customer segmentation based on RFM and behavioural features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from features.customer_features import compute_behavioural_features, compute_rfm
from models.model_utils import MODELS_DIR, save_json, save_pickle


@dataclass
class CustomerSegmentationSummary:
    model_name: str
    trained_at: str
    customer_count: int
    segment_count: int
    largest_segment: str


class CustomerSegmentationModel:
    """Cluster customers into stable, explainable commercial segments."""

    SEGMENT_NAMES = [
        "Champions",
        "Loyal",
        "Growth",
        "At Risk",
        "Dormant",
        "Reactivation",
    ]

    def __init__(self, artifact_dir: Path | None = None) -> None:
        self.artifact_dir = artifact_dir or MODELS_DIR
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_path = self.artifact_dir / "customer_segmentation.pkl"
        self.assignments_path = self.artifact_dir / "customer_segments.parquet"
        self.summary_path = self.artifact_dir / "customer_segmentation_summary.json"

    def _prepare_features(
        self,
        transactions_df: pd.DataFrame,
        crm_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        rfm = compute_rfm(transactions_df)
        behavioural = compute_behavioural_features(transactions_df)
        feature_df = rfm.merge(behavioural, on="customer_id", how="left")
        if crm_df is not None and "customer_id" in crm_df.columns:
            feature_df = feature_df.merge(crm_df, on="customer_id", how="left")
        feature_df["days_between_orders"] = feature_df["days_between_orders"].fillna(feature_df["days_between_orders"].median())
        feature_df["avg_basket_size"] = feature_df["avg_basket_size"].fillna(feature_df["avg_basket_size"].median())
        feature_df["category_breadth"] = feature_df["category_breadth"].fillna(0)
        return feature_df

    def train(
        self,
        transactions_df: pd.DataFrame,
        crm_df: pd.DataFrame | None = None,
        n_clusters: int = 5,
    ) -> dict[str, object]:
        feature_df = self._prepare_features(transactions_df, crm_df)
        numeric_columns = [
            "recency_days",
            "frequency",
            "monetary_value",
            "avg_order_value",
            "avg_basket_size",
            "category_breadth",
            "days_between_orders",
            "customer_lifetime_value_gbp",
            "total_orders",
        ]
        available_numeric = [column for column in numeric_columns if column in feature_df.columns]
        training_matrix = feature_df[available_numeric].fillna(feature_df[available_numeric].median())

        scaler = StandardScaler()
        scaled = scaler.fit_transform(training_matrix)
        model = KMeans(n_clusters=min(n_clusters, max(len(feature_df), 1)), random_state=42, n_init=10)
        feature_df["segment_id"] = model.fit_predict(scaled)

        profiles = (
            feature_df.groupby("segment_id", as_index=False)[["recency_days", "frequency", "monetary_value"]]
            .mean()
            .sort_values(["monetary_value", "frequency", "recency_days"], ascending=[False, False, True])
        )
        ordered_ids = profiles["segment_id"].tolist()
        segment_name_map = {
            segment_id: self.SEGMENT_NAMES[index] if index < len(self.SEGMENT_NAMES) else f"Segment {index + 1}"
            for index, segment_id in enumerate(ordered_ids)
        }
        feature_df["segment_name"] = feature_df["segment_id"].map(segment_name_map)

        segment_summary = (
            feature_df.groupby("segment_name", as_index=False)
            .agg(
                customers=("customer_id", "count"),
                avg_recency_days=("recency_days", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary_value=("monetary_value", "mean"),
            )
            .sort_values("customers", ascending=False)
        )
        summary = CustomerSegmentationSummary(
            model_name="customer_segmentation",
            trained_at=datetime.utcnow().isoformat(),
            customer_count=int(feature_df["customer_id"].nunique()),
            segment_count=int(segment_summary["segment_name"].nunique()),
            largest_segment=str(segment_summary.iloc[0]["segment_name"]),
        )

        artifact = {
            "scaler": scaler,
            "model": model,
            "numeric_columns": available_numeric,
            "segment_name_map": segment_name_map,
            "summary": summary,
        }
        save_pickle(artifact, self.artifact_path)
        feature_df.to_parquet(self.assignments_path, index=False)
        save_json(
            {
                "summary": summary,
                "segments": segment_summary,
            },
            self.summary_path,
        )
        return {
            "artifact_path": self.artifact_path,
            "segments": feature_df,
            "segment_summary": segment_summary,
            "summary": summary,
        }