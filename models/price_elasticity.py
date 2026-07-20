"""Price elasticity of demand by category.

Estimates the price elasticity of demand from observed promotional price
variation using a log-log fixed-effects regression: within each category we
regress log(weekly units) on log(effective price) after removing per-SKU means
(SKU fixed effects), so the coefficient is identified from within-SKU price
moves (promotions), not cross-SKU differences. Effective price is the list price
net of the promotional discount.

Each category ends in a decision: whether it is elastic enough to promote, and
the deepest discount that still protects unit margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from models.model_utils import MODELS_DIR, save_json, save_pickle

MIN_OBS = 50
MAX_RECOMMENDED_DISCOUNT = 0.40
MARGIN_FLOOR = 0.10  # keep at least 10% gross margin after discount


@dataclass
class PriceElasticitySummary:
    model_name: str
    trained_at: str
    category_count: int
    mean_elasticity: float
    elastic_category_count: int  # CI upper bound below -1


class PriceElasticityModel:
    """Estimate category price elasticities and safe discount depths."""

    def __init__(self, artifact_dir: Path | None = None) -> None:
        self.artifact_dir = artifact_dir or MODELS_DIR
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_path = self.artifact_dir / "price_elasticity.pkl"
        self.table_path = self.artifact_dir / "price_elasticity.parquet"
        self.summary_path = self.artifact_dir / "price_elasticity_summary.json"

    def _weekly_panel(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        tx = transactions_df.copy()
        tx = tx[tx["quantity"] > 0].copy()
        tx["invoice_date"] = pd.to_datetime(tx["invoice_date"])
        discount = tx["discount_pct"] if "discount_pct" in tx.columns else 0.0
        tx["effective_price"] = tx["unit_price_gbp"].astype(float) * (1.0 - discount)
        tx["week"] = tx["invoice_date"].dt.to_period("W").dt.start_time
        panel = (
            tx.groupby(["category", "stock_code", "week"], as_index=False)
            .agg(units=("quantity", "sum"), effective_price=("effective_price", "mean"))
        )
        return panel[(panel["units"] > 0) & (panel["effective_price"] > 0)]

    def _estimate_category(self, frame: pd.DataFrame) -> dict[str, float] | None:
        frame = frame.copy()
        frame["log_units"] = np.log(frame["units"])
        frame["log_price"] = np.log(frame["effective_price"])
        # SKU fixed effects: demean within stock_code.
        frame["lp_demeaned"] = frame["log_price"] - frame.groupby("stock_code")["log_price"].transform("mean")
        frame["ly_demeaned"] = frame["log_units"] - frame.groupby("stock_code")["log_units"].transform("mean")
        if len(frame) < MIN_OBS or frame["lp_demeaned"].std() == 0:
            return None
        design = sm.add_constant(frame["lp_demeaned"].to_numpy())
        model = sm.OLS(frame["ly_demeaned"].to_numpy(), design).fit()
        elasticity = float(model.params[1])
        ci_low, ci_high = (float(x) for x in model.conf_int()[1])
        return {
            "elasticity": elasticity,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": float(model.pvalues[1]),
            "n_obs": int(len(frame)),
        }

    def _safe_discount_depth(self, category: str, inventory_df: pd.DataFrame | None) -> float:
        """Deepest discount that keeps gross margin above the floor."""
        if inventory_df is None or "category" not in inventory_df.columns:
            return MAX_RECOMMENDED_DISCOUNT
        rows = inventory_df[inventory_df["category"] == category]
        if rows.empty or "unit_cost_gbp" not in rows.columns:
            return MAX_RECOMMENDED_DISCOUNT
        price = float(rows["unit_price_gbp"].mean())
        cost = float(rows["unit_cost_gbp"].mean())
        if price <= 0:
            return MAX_RECOMMENDED_DISCOUNT
        # discounted price must satisfy (p*(1-d) - c)/(p*(1-d)) >= MARGIN_FLOOR
        # => p*(1-d) >= c / (1 - MARGIN_FLOOR)
        min_price = cost / (1.0 - MARGIN_FLOOR)
        max_discount = 1.0 - min_price / price
        return float(np.clip(min(max_discount, MAX_RECOMMENDED_DISCOUNT), 0.0, MAX_RECOMMENDED_DISCOUNT))

    def train(
        self,
        transactions_df: pd.DataFrame,
        inventory_df: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        panel = self._weekly_panel(transactions_df)
        if panel.empty:
            raise ValueError("No positive-quantity transactions available for elasticity estimation.")

        rows: list[dict[str, object]] = []
        for category, frame in panel.groupby("category"):
            estimate = self._estimate_category(frame)
            if estimate is None:
                continue
            is_elastic = estimate["ci_high"] < -1.0
            safe_discount = self._safe_discount_depth(str(category), inventory_df)
            if is_elastic:
                decision = f"Promote: elastic demand, discount up to {safe_discount:.0%} (margin-protected)"
            elif estimate["ci_low"] > -1.0:
                decision = "Hold price: inelastic demand, discounting erodes revenue"
            else:
                decision = "Test cautiously: elasticity not distinguishable from -1"
            rows.append(
                {
                    "category": str(category),
                    "elasticity": round(estimate["elasticity"], 3),
                    "ci_low": round(estimate["ci_low"], 3),
                    "ci_high": round(estimate["ci_high"], 3),
                    "p_value": round(estimate["p_value"], 5),
                    "n_obs": estimate["n_obs"],
                    "is_elastic": is_elastic,
                    "recommended_max_discount": round(safe_discount, 3),
                    "recommended_action": decision,
                }
            )

        table = pd.DataFrame(rows).sort_values("elasticity")
        summary = PriceElasticitySummary(
            model_name="price_elasticity",
            trained_at=datetime.utcnow().isoformat(),
            category_count=int(len(table)),
            mean_elasticity=round(float(table["elasticity"].mean()), 3) if not table.empty else 0.0,
            elastic_category_count=int(table["is_elastic"].sum()) if not table.empty else 0,
        )

        save_pickle({"table": table, "summary": summary}, self.artifact_path)
        table.to_parquet(self.table_path, index=False)
        save_json({"summary": summary, "elasticities": table}, self.summary_path)
        return {"artifact_path": self.artifact_path, "table": table, "summary": summary}
