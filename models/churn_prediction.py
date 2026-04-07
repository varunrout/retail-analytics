"""Customer churn prediction model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features.customer_features import compute_behavioural_features, compute_rfm
from models.model_utils import MODELS_DIR, risk_band_from_probability, save_json, save_pickle


@dataclass
class ChurnPredictionSummary:
    model_name: str
    trained_at: str
    customer_count: int
    positive_rate: float
    roc_auc: float
    accuracy: float


class ChurnPredictionModel:
    """Predict customer churn risk using behavioural and CRM signals."""

    def __init__(self, artifact_dir: Path | None = None) -> None:
        self.artifact_dir = artifact_dir or MODELS_DIR
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_path = self.artifact_dir / "churn_prediction.pkl"
        self.scored_path = self.artifact_dir / "customer_churn_scores.parquet"
        self.summary_path = self.artifact_dir / "churn_prediction_summary.json"

    def _prepare_frame(
        self,
        transactions_df: pd.DataFrame,
        crm_df: pd.DataFrame | None = None,
        campaigns_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        rfm = compute_rfm(transactions_df)
        behavioural = compute_behavioural_features(transactions_df)
        frame = rfm.merge(behavioural, on="customer_id", how="left")
        if crm_df is not None and "customer_id" in crm_df.columns:
            frame = frame.merge(crm_df, on="customer_id", how="left")
        if campaigns_df is not None and "customer_id" in campaigns_df.columns:
            engagement = (
                campaigns_df.groupby("customer_id", as_index=False)
                .agg(
                    email_open_rate=("open_flag", "mean"),
                    email_click_rate=("click_flag", "mean"),
                    conversion_rate=("conversion_flag", "mean"),
                )
            )
            frame = frame.merge(engagement, on="customer_id", how="left")

        for column in ["email_open_rate", "email_click_rate", "conversion_rate", "days_between_orders", "avg_basket_size", "category_breadth"]:
            if column in frame.columns:
                frame[column] = frame[column].fillna(frame[column].median())

        if "is_active" in frame.columns:
            frame["churned"] = (~frame["is_active"].fillna(False)).astype(int)
        else:
            frame["churned"] = (frame["recency_days"] > 180).astype(int)

        class_counts = frame["churned"].value_counts()
        if frame["churned"].nunique() == 1 or class_counts.min() < 10:
            threshold = frame["recency_days"].quantile(0.75)
            frame["churned"] = (
                (frame["recency_days"] >= threshold)
                | (frame.get("purchase_frequency_trend", "stable") == "declining")
            ).astype(int)

        if frame["churned"].nunique() == 1:
            frame["churned"] = (frame["recency_days"] > frame["recency_days"].median()).astype(int)
        return frame

    def train(
        self,
        transactions_df: pd.DataFrame,
        crm_df: pd.DataFrame | None = None,
        campaigns_df: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        frame = self._prepare_frame(transactions_df, crm_df, campaigns_df)
        numeric_columns = [
            column
            for column in [
                "recency_days",
                "frequency",
                "monetary_value",
                "avg_order_value",
                "avg_basket_size",
                "category_breadth",
                "days_between_orders",
                "customer_lifetime_value_gbp",
                "total_orders",
                "email_open_rate",
                "email_click_rate",
                "conversion_rate",
            ]
            if column in frame.columns
        ]
        categorical_columns = [
            column
            for column in [
                "channel_preference",
                "purchase_frequency_trend",
                "preferred_season",
                "loyalty_tier",
                "age_band",
                "gender",
                "acquisition_channel",
                "preferred_category",
            ]
            if column in frame.columns
        ]

        training_frame = frame[numeric_columns + categorical_columns + ["churned"]].copy()
        for column in categorical_columns:
            training_frame[column] = training_frame[column].fillna("Unknown")
        for column in numeric_columns:
            training_frame[column] = training_frame[column].fillna(training_frame[column].median())

        X = training_frame.drop(columns=["churned"])
        y = training_frame["churned"]
        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numeric_columns),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ]
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        pipeline.fit(X_train, y_train)

        test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= 0.5).astype(int)
        roc_auc = float(roc_auc_score(y_test, test_proba)) if y_test.nunique() > 1 else 0.5
        accuracy = float(accuracy_score(y_test, test_pred))

        frame["churn_probability"] = pipeline.predict_proba(X)[:, 1]
        frame["risk_band"] = frame["churn_probability"].apply(risk_band_from_probability)

        classifier = pipeline.named_steps["classifier"]
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        coefficient_table = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "coefficient": classifier.coef_[0],
                }
            )
            .assign(abs_coefficient=lambda data: data["coefficient"].abs())
            .sort_values("abs_coefficient", ascending=False)
            .drop(columns=["abs_coefficient"])
        )
        summary = ChurnPredictionSummary(
            model_name="churn_prediction",
            trained_at=datetime.utcnow().isoformat(),
            customer_count=int(frame["customer_id"].nunique()),
            positive_rate=round(float(frame["churned"].mean()), 4),
            roc_auc=round(roc_auc, 4),
            accuracy=round(accuracy, 4),
        )

        artifact = {
            "pipeline": pipeline,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "summary": summary,
        }
        save_pickle(artifact, self.artifact_path)
        frame.to_parquet(self.scored_path, index=False)
        save_json(
            {
                "summary": summary,
                "top_drivers": coefficient_table.head(15),
            },
            self.summary_path,
        )
        return {
            "artifact_path": self.artifact_path,
            "scores": frame,
            "coefficients": coefficient_table,
            "summary": summary,
        }