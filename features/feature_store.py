"""
Feature store: combines all feature sets into a unified feature matrix.
Saves feature matrix to data/features/ with metadata.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional
import json
import logging

from features.calendar_features import add_calendar_features
from features.product_features import compute_product_features, compute_price_features
from features.demand_features import compute_rolling_demand, flag_sparse_demand
from features.inventory_features import compute_inventory_features
from features.customer_features import compute_rfm, compute_behavioural_features

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for a single feature."""

    name: str
    description: str
    formula: str
    feature_type: Literal["numeric", "categorical", "boolean", "datetime"]
    leakage_risk: Literal["none", "low", "high"]
    refresh_cadence: Literal["daily", "weekly", "monthly"]
    model_usage: list
    source_table: str


class FeatureStore:
    """Central feature store combining all feature sets."""

    FEATURE_REGISTRY: list = [
        FeatureMetadata("total_units_sold_12m", "Units sold in last 12 months", "sum(quantity, 12m)", "numeric", "none", "monthly", ["demand_forecast", "inventory_scoring"], "transactions"),
        FeatureMetadata("total_revenue_12m", "Net revenue in last 12 months", "sum(net_revenue_gbp, 12m)", "numeric", "none", "monthly", ["demand_forecast", "customer_segmentation"], "transactions"),
        FeatureMetadata("avg_unit_price", "Average selling price", "mean(unit_price_gbp)", "numeric", "none", "weekly", ["demand_forecast", "churn_prediction"], "transactions"),
        FeatureMetadata("price_cv", "Price coefficient of variation", "std(price)/mean(price)", "numeric", "none", "weekly", ["demand_forecast"], "transactions"),
        FeatureMetadata("return_rate", "Proportion of orders returned", "returns/total_orders", "numeric", "none", "monthly", ["inventory_scoring"], "transactions"),
        FeatureMetadata("is_seasonal", "Flag: seasonal demand pattern", "monthly_cv > 0.3", "boolean", "none", "monthly", ["demand_forecast"], "transactions"),
        FeatureMetadata("rolling_28d_units", "Rolling 28-day unit sales", "sum(quantity, 28d)", "numeric", "low", "daily", ["demand_forecast"], "transactions"),
        FeatureMetadata("rolling_28d_cv", "Rolling 28-day demand CV", "std/mean over 28d", "numeric", "low", "daily", ["demand_forecast", "inventory_scoring"], "transactions"),
        FeatureMetadata("abc_class", "ABC revenue classification", "cumulative revenue bands", "categorical", "none", "monthly", ["inventory_scoring"], "transactions"),
        FeatureMetadata("xyz_class", "XYZ demand variability class", "CV of weekly sales", "categorical", "none", "weekly", ["inventory_scoring", "demand_forecast"], "transactions"),
        FeatureMetadata("days_cover", "Days of stock at current demand", "stock_on_hand / avg_daily_demand", "numeric", "none", "daily", ["inventory_scoring"], "inventory"),
        FeatureMetadata("stockout_probability", "Probability of stockout in lead time", "P(D_LT > SOH)", "numeric", "none", "daily", ["inventory_scoring"], "inventory"),
        FeatureMetadata("recency_days", "Days since last purchase", "ref_date - last_purchase_date", "numeric", "none", "daily", ["customer_segmentation", "churn_prediction"], "transactions"),
        FeatureMetadata("frequency", "Number of orders in 12m", "count(distinct invoice_date)", "numeric", "none", "monthly", ["customer_segmentation", "churn_prediction"], "transactions"),
        FeatureMetadata("monetary_value", "Total spend in 12m", "sum(net_revenue_gbp)", "numeric", "none", "monthly", ["customer_segmentation", "churn_prediction"], "transactions"),
        FeatureMetadata("r_score", "Recency quintile score 1-5", "qcut(recency_days, 5, inverted)", "numeric", "none", "monthly", ["customer_segmentation", "churn_prediction"], "transactions"),
        FeatureMetadata("f_score", "Frequency quintile score 1-5", "qcut(frequency, 5)", "numeric", "none", "monthly", ["customer_segmentation", "churn_prediction"], "transactions"),
        FeatureMetadata("m_score", "Monetary quintile score 1-5", "qcut(monetary_value, 5)", "numeric", "none", "monthly", ["customer_segmentation", "churn_prediction"], "transactions"),
        FeatureMetadata("days_between_orders", "Avg days between orders", "mean(diff(order_dates))", "numeric", "none", "monthly", ["churn_prediction"], "transactions"),
        FeatureMetadata("category_breadth", "Distinct L1 categories purchased", "nunique(category_l1)", "numeric", "none", "monthly", ["customer_segmentation", "churn_prediction"], "transactions"),
        FeatureMetadata("price_band", "Price band classification", "avg_price brackets", "categorical", "none", "monthly", ["customer_segmentation"], "transactions"),
        FeatureMetadata("month_sin", "Cyclic month encoding (sin)", "sin(2*pi*month/12)", "numeric", "none", "daily", ["demand_forecast"], "calendar"),
        FeatureMetadata("month_cos", "Cyclic month encoding (cos)", "cos(2*pi*month/12)", "numeric", "none", "daily", ["demand_forecast"], "calendar"),
    ]

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")
        self.features_dir = self.data_dir / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def build_product_feature_matrix(
        self,
        transactions_df: pd.DataFrame,
        inventory_df: Optional[pd.DataFrame] = None,
        costs_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build per-SKU feature matrix for demand forecasting and inventory scoring."""
        self.logger.info("Building product feature matrix...")

        product_feats = compute_product_features(transactions_df, inventory_df, costs_df)
        price_feats = compute_price_features(transactions_df)

        matrix = product_feats.join(
            price_feats[["price_band", "price_elasticity_estimate", "promo_frequency", "avg_discount_pct"]],
            how="left",
        )

        sku_col = "stock_code" if transactions_df.columns.__contains__("stock_code") else "sku"
        sparse_flags = flag_sparse_demand(
            transactions_df, "quantity", "invoice_date", sku_col
        )
        matrix = matrix.join(sparse_flags.rename("is_sparse"), how="left")
        matrix["is_sparse"] = matrix["is_sparse"].fillna(False)

        if inventory_df is not None:
            inv_feats = compute_inventory_features(inventory_df, transactions_df)
            inv_index_col = sku_col
            if inv_index_col in inv_feats.columns:
                inv_feats = inv_feats.set_index(inv_index_col)
            keep_cols = [c for c in ["days_cover", "stockout_probability", "abc_class", "xyz_class"] if c in inv_feats.columns]
            if keep_cols:
                matrix = matrix.join(inv_feats[keep_cols], how="left")

        self.logger.info(f"Product feature matrix: {matrix.shape}")
        return matrix

    def build_customer_feature_matrix(
        self,
        transactions_df: pd.DataFrame,
        crm_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build per-customer feature matrix for segmentation and churn prediction."""
        self.logger.info("Building customer feature matrix...")

        rfm = compute_rfm(transactions_df)
        behavioural = compute_behavioural_features(transactions_df)

        matrix = rfm.merge(behavioural, on="customer_id", how="left")

        if crm_df is not None and "customer_id" in crm_df.columns:
            matrix = matrix.merge(crm_df, on="customer_id", how="left")

        self.logger.info(f"Customer feature matrix: {matrix.shape}")
        return matrix

    def save_feature_matrix(self, df: pd.DataFrame, name: str) -> Path:
        """Save feature matrix as parquet with metadata sidecar JSON."""
        parquet_path = self.features_dir / f"{name}.parquet"
        df.to_parquet(parquet_path, index=True)

        meta = {
            "name": name,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "created_at": pd.Timestamp.now().isoformat(),
        }
        meta_path = self.features_dir / f"{name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self.logger.info(f"Saved feature matrix '{name}' to {parquet_path}")
        return parquet_path

    def load_feature_matrix(self, name: str) -> pd.DataFrame:
        """Load previously saved feature matrix."""
        parquet_path = self.features_dir / f"{name}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Feature matrix '{name}' not found at {parquet_path}")
        return pd.read_parquet(parquet_path)

    def get_feature_catalog(self) -> pd.DataFrame:
        """Return feature registry as DataFrame."""
        records = []
        for feat in self.FEATURE_REGISTRY:
            records.append(
                {
                    "name": feat.name,
                    "description": feat.description,
                    "formula": feat.formula,
                    "feature_type": feat.feature_type,
                    "leakage_risk": feat.leakage_risk,
                    "refresh_cadence": feat.refresh_cadence,
                    "model_usage": ", ".join(feat.model_usage),
                    "source_table": feat.source_table,
                }
            )
        return pd.DataFrame(records)

    def run(
        self,
        transactions_df: Optional[pd.DataFrame] = None,
        inventory_df: Optional[pd.DataFrame] = None,
        costs_df: Optional[pd.DataFrame] = None,
        crm_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Build all feature matrices.
        If no DataFrames provided, loads from data/synthetic/ directory.
        Returns dict of {matrix_name: DataFrame}.
        """
        if transactions_df is None:
            synthetic_dir = self.data_dir / "synthetic"
            txn_path = synthetic_dir / "transactions.parquet"
            if txn_path.exists():
                transactions_df = pd.read_parquet(txn_path)
            else:
                self.logger.warning("No transactions data found. Using demo data.")
                from models.model_utils import make_demo_transactions
                transactions_df = make_demo_transactions()

        if inventory_df is None:
            synthetic_dir = self.data_dir / "synthetic"
            inv_path = synthetic_dir / "inventory.parquet"
            if inv_path.exists():
                inventory_df = pd.read_parquet(inv_path)

        matrices = {}

        product_matrix = self.build_product_feature_matrix(transactions_df, inventory_df, costs_df)
        self.save_feature_matrix(product_matrix, "product_features")
        matrices["product_features"] = product_matrix

        if "customer_id" in transactions_df.columns:
            customer_matrix = self.build_customer_feature_matrix(transactions_df, crm_df)
            self.save_feature_matrix(customer_matrix, "customer_features")
            matrices["customer_features"] = customer_matrix

        return matrices


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = FeatureStore()
    matrices = store.run()
    for name, df in matrices.items():
        print(f"{name}: {df.shape}")
