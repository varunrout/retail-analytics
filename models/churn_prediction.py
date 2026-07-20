"""Customer churn prediction model.

Churn is defined on a FORWARD window to avoid label leakage: given transactions
up to a cut-off date, a customer is 'churned' if they made no purchase in the
following ``horizon_days``. All features are computed strictly from data on or
before the cut-off, so recency at the cut-off is a legitimate predictor rather
than a restatement of the label (the previous version derived the label from
``recency_days`` while also using it as a feature, which is why ROC-AUC was 0.499).

The model is evaluated against two baselines - a majority-class predictor and a
recency-rule predictor - plus a calibration curve, so its ROC-AUC/PR-AUC are
reported next to a reference, not in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features.customer_features import compute_behavioural_features, compute_rfm
from models.model_utils import MODELS_DIR, risk_band_from_probability, save_json, save_pickle

DEFAULT_HORIZON_DAYS = 90

# Recommended retention action per risk band - the decision, not just a score.
RETENTION_ACTION = {
    "High": "Priority win-back: personal offer + service call",
    "Medium": "Automated re-engagement email with incentive",
    "Low": "Standard lifecycle marketing",
}


@dataclass
class ChurnPredictionSummary:
    model_name: str
    trained_at: str
    customer_count: int
    horizon_days: int
    positive_rate: float
    roc_auc: float
    pr_auc: float
    baseline_majority_auc: float
    baseline_recency_auc: float
    baseline_majority_pr_auc: float
    accuracy: float
    calibration_mae: float


class ChurnPredictionModel:
    """Predict customer churn risk from behaviour observed before a cut-off."""

    NUMERIC_CANDIDATES = [
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
    CATEGORICAL_CANDIDATES = [
        "channel_preference",
        "purchase_frequency_trend",
        "preferred_season",
        "loyalty_tier",
        "age_band",
        "gender",
        "acquisition_channel",
        "preferred_category",
    ]

    def __init__(self, artifact_dir: Path | None = None) -> None:
        self.artifact_dir = artifact_dir or MODELS_DIR
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_path = self.artifact_dir / "churn_prediction.pkl"
        self.scored_path = self.artifact_dir / "customer_churn_scores.parquet"
        self.summary_path = self.artifact_dir / "churn_prediction_summary.json"

    def _feature_frame(
        self,
        transactions_df: pd.DataFrame,
        reference_date: pd.Timestamp,
        crm_df: pd.DataFrame | None,
        campaigns_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Build customer features using only transactions on/before reference_date."""
        tx = transactions_df.copy()
        tx["invoice_date"] = pd.to_datetime(tx["invoice_date"])
        tx = tx[tx["invoice_date"] <= reference_date]

        rfm = compute_rfm(tx, reference_date=reference_date.date())
        behavioural = compute_behavioural_features(tx, reference_date=reference_date.date())
        frame = rfm.merge(behavioural, on="customer_id", how="left")

        if crm_df is not None and "customer_id" in crm_df.columns:
            # Drop CRM columns that would leak future/label-adjacent state.
            crm = crm_df.drop(columns=[c for c in ["is_active", "last_purchase_date"] if c in crm_df.columns])
            frame = frame.merge(crm, on="customer_id", how="left")
        if campaigns_df is not None and "customer_id" in campaigns_df.columns:
            campaigns = campaigns_df.copy()
            if "contact_date" in campaigns.columns:
                campaigns["contact_date"] = pd.to_datetime(campaigns["contact_date"])
                campaigns = campaigns[campaigns["contact_date"] <= reference_date]
            engagement = campaigns.groupby("customer_id", as_index=False).agg(
                email_open_rate=("open_flag", "mean"),
                email_click_rate=("click_flag", "mean"),
                conversion_rate=("conversion_flag", "mean"),
            )
            frame = frame.merge(engagement, on="customer_id", how="left")
        return frame

    def _forward_label(
        self,
        transactions_df: pd.DataFrame,
        reference_date: pd.Timestamp,
        horizon_days: int,
    ) -> set[str]:
        """Customers who purchased in (reference_date, reference_date + horizon]."""
        tx = transactions_df.copy()
        tx["invoice_date"] = pd.to_datetime(tx["invoice_date"])
        window = tx[
            (tx["invoice_date"] > reference_date)
            & (tx["invoice_date"] <= reference_date + pd.Timedelta(days=horizon_days))
        ]
        return set(window["customer_id"].unique())

    def _columns(self, frame: pd.DataFrame) -> tuple[list[str], list[str]]:
        numeric = [c for c in self.NUMERIC_CANDIDATES if c in frame.columns]
        categorical = [c for c in self.CATEGORICAL_CANDIDATES if c in frame.columns]
        return numeric, categorical

    def _prepare_design(self, frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
        design = frame[numeric + categorical].copy()
        for column in categorical:
            design[column] = design[column].fillna("Unknown")
        for column in numeric:
            design[column] = design[column].fillna(design[column].median())
        return design

    def train(
        self,
        transactions_df: pd.DataFrame,
        crm_df: pd.DataFrame | None = None,
        campaigns_df: pd.DataFrame | None = None,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
    ) -> dict[str, object]:
        tx = transactions_df.copy()
        tx["invoice_date"] = pd.to_datetime(tx["invoice_date"])
        max_date = tx["invoice_date"].max()
        cutoff = max_date - pd.Timedelta(days=horizon_days)

        # --- Honest evaluation frame: features <= cutoff, label from forward window ---
        eval_frame = self._feature_frame(tx, cutoff, crm_df, campaigns_df)
        purchasers_future = self._forward_label(tx, cutoff, horizon_days)
        eval_frame = eval_frame[eval_frame["customer_id"].notna()].copy()
        eval_frame["churned"] = (~eval_frame["customer_id"].isin(purchasers_future)).astype(int)

        numeric, categorical = self._columns(eval_frame)
        X = self._prepare_design(eval_frame, numeric, categorical)
        y = eval_frame["churned"]

        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ]
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        )
        pipeline.fit(X_train, y_train)
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= 0.5).astype(int)

        two_class = y_test.nunique() > 1
        base_rate = float(y_test.mean())
        roc_auc = float(roc_auc_score(y_test, test_proba)) if two_class else float("nan")
        pr_auc = float(average_precision_score(y_test, test_proba)) if two_class else float("nan")
        accuracy = float((test_pred == y_test).mean())

        # Baselines on the identical test set.
        baseline_majority_auc = 0.5
        baseline_majority_pr_auc = base_rate
        if two_class and "recency_days" in X_test.columns:
            baseline_recency_auc = float(roc_auc_score(y_test, X_test["recency_days"]))
        else:
            baseline_recency_auc = float("nan")

        # Calibration: mean abs gap between predicted and observed churn per decile.
        calibration = self._calibration(y_test.to_numpy(), test_proba)
        calibration_mae = (
            float(np.mean([abs(b["predicted"] - b["observed"]) for b in calibration]))
            if calibration
            else float("nan")
        )

        # --- Serving scores: score everyone at the latest reference date ---
        score_frame = self._feature_frame(tx, max_date, crm_df, campaigns_df)
        score_frame = score_frame[score_frame["customer_id"].notna()].copy()
        score_numeric, score_categorical = self._columns(score_frame)
        score_design = self._prepare_design(score_frame, score_numeric, score_categorical)
        # Align columns to the trained pipeline.
        for col in numeric + categorical:
            if col not in score_design.columns:
                score_design[col] = "Unknown" if col in categorical else 0.0
        score_design = score_design[numeric + categorical]
        score_frame["churn_probability"] = pipeline.predict_proba(score_design)[:, 1]
        score_frame["risk_band"] = score_frame["churn_probability"].apply(risk_band_from_probability)
        score_frame["recommended_action"] = score_frame["risk_band"].map(RETENTION_ACTION)

        classifier = pipeline.named_steps["classifier"]
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        coefficient_table = (
            pd.DataFrame({"feature": feature_names, "coefficient": classifier.coef_[0]})
            .assign(abs_coefficient=lambda data: data["coefficient"].abs())
            .sort_values("abs_coefficient", ascending=False)
            .drop(columns=["abs_coefficient"])
        )

        summary = ChurnPredictionSummary(
            model_name="churn_prediction",
            trained_at=datetime.utcnow().isoformat(),
            customer_count=int(eval_frame["customer_id"].nunique()),
            horizon_days=horizon_days,
            positive_rate=round(float(eval_frame["churned"].mean()), 4),
            roc_auc=round(roc_auc, 4),
            pr_auc=round(pr_auc, 4),
            baseline_majority_auc=baseline_majority_auc,
            baseline_recency_auc=round(baseline_recency_auc, 4),
            baseline_majority_pr_auc=round(baseline_majority_pr_auc, 4),
            accuracy=round(accuracy, 4),
            calibration_mae=round(calibration_mae, 4),
        )

        artifact = {
            "pipeline": pipeline,
            "numeric_columns": numeric,
            "categorical_columns": categorical,
            "horizon_days": horizon_days,
            "summary": summary,
        }
        save_pickle(artifact, self.artifact_path)
        score_frame.to_parquet(self.scored_path, index=False)
        save_json(
            {"summary": summary, "top_drivers": coefficient_table.head(15), "calibration": calibration},
            self.summary_path,
        )
        return {
            "artifact_path": self.artifact_path,
            "scores": score_frame,
            "coefficients": coefficient_table,
            "summary": summary,
        }

    @staticmethod
    def _calibration(y_true: np.ndarray, proba: np.ndarray, bins: int = 5) -> list[dict[str, float]]:
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            return []
        order = np.argsort(proba)
        y_true, proba = y_true[order], proba[order]
        chunks = np.array_split(np.arange(len(proba)), bins)
        out: list[dict[str, float]] = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            out.append(
                {
                    "predicted": round(float(proba[chunk].mean()), 4),
                    "observed": round(float(y_true[chunk].mean()), 4),
                    "n": int(len(chunk)),
                }
            )
        return out
