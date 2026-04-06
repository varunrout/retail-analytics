"""
Model drift detection and monitoring.
Monitors feature distributions and prediction distributions for drift.
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    feature_name: str
    drift_detected: bool
    drift_score: float
    method: str  # ks_test / psi / chi_squared
    p_value: Optional[float]
    threshold: float
    baseline_stats: dict
    current_stats: dict


class ModelMonitor:
    """Monitors ML model health via feature and prediction drift."""

    def __init__(self, baseline_dir: Path = Path("data/model_baselines")) -> None:
        self.baseline_dir = baseline_dir
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Statistical methods
    # ------------------------------------------------------------------

    def compute_psi(
        self,
        baseline: pd.Series,
        current: pd.Series,
        n_bins: int = 10,
    ) -> float:
        """
        Population Stability Index (PSI).
        PSI < 0.1: no significant change
        0.1 <= PSI < 0.25: moderate change
        PSI >= 0.25: significant change (model retraining needed)
        """
        baseline_clean = baseline.dropna()
        current_clean = current.dropna()
        if baseline_clean.empty or current_clean.empty:
            return 0.0

        bin_edges = np.percentile(baseline_clean, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        base_counts, _ = np.histogram(baseline_clean, bins=bin_edges)
        curr_counts, _ = np.histogram(current_clean, bins=bin_edges)

        eps = 1e-6
        base_pct = (base_counts + eps) / (len(baseline_clean) + eps * len(base_counts))
        curr_pct = (curr_counts + eps) / (len(current_clean) + eps * len(curr_counts))

        psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
        return round(psi, 6)

    def ks_test(
        self,
        baseline: pd.Series,
        current: pd.Series,
    ) -> tuple[float, float]:
        """Kolmogorov-Smirnov test. Returns (statistic, p_value)."""
        b = baseline.dropna().values
        c = current.dropna().values
        if len(b) == 0 or len(c) == 0:
            return 0.0, 1.0
        result = stats.ks_2samp(b, c)
        return float(result.statistic), float(result.pvalue)

    # ------------------------------------------------------------------
    # Drift checks
    # ------------------------------------------------------------------

    def check_feature_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: list[str],
    ) -> list[DriftResult]:
        """Check all features for drift. Numerical: KS test + PSI. Categorical: chi-squared."""
        results: list[DriftResult] = []
        for feat in features:
            if feat not in baseline_df.columns or feat not in current_df.columns:
                logger.warning("Feature '%s' missing from one of the DataFrames; skipping.", feat)
                continue

            base_series = baseline_df[feat]
            curr_series = current_df[feat]

            if pd.api.types.is_numeric_dtype(base_series):
                ks_stat, p_val = self.ks_test(base_series, curr_series)
                psi = self.compute_psi(base_series, curr_series)
                threshold = 0.05  # p-value threshold
                drift_detected = (p_val < threshold) or (psi >= 0.25)
                results.append(
                    DriftResult(
                        feature_name=feat,
                        drift_detected=drift_detected,
                        drift_score=round(max(ks_stat, psi), 6),
                        method="ks_test",
                        p_value=round(p_val, 6),
                        threshold=threshold,
                        baseline_stats={
                            "mean": float(base_series.mean()),
                            "std": float(base_series.std()),
                            "psi": psi,
                        },
                        current_stats={
                            "mean": float(curr_series.mean()),
                            "std": float(curr_series.std()),
                        },
                    )
                )
            else:
                # Categorical: chi-squared
                all_cats = set(base_series.dropna().unique()) | set(curr_series.dropna().unique())
                base_counts = base_series.value_counts().reindex(list(all_cats), fill_value=0)
                curr_counts = curr_series.value_counts().reindex(list(all_cats), fill_value=0)
                if base_counts.sum() == 0 or curr_counts.sum() == 0:
                    continue
                chi2, p_val, *_ = stats.chi2_contingency(
                    np.array([base_counts.values, curr_counts.values])
                )
                threshold = 0.05
                results.append(
                    DriftResult(
                        feature_name=feat,
                        drift_detected=bool(p_val < threshold),
                        drift_score=round(float(chi2), 4),
                        method="chi_squared",
                        p_value=round(float(p_val), 6),
                        threshold=threshold,
                        baseline_stats=base_counts.to_dict(),
                        current_stats=curr_counts.to_dict(),
                    )
                )
        return results

    def check_prediction_drift(
        self,
        baseline_preds: pd.Series,
        current_preds: pd.Series,
    ) -> DriftResult:
        """Check if prediction distribution has shifted."""
        ks_stat, p_val = self.ks_test(baseline_preds, current_preds)
        psi = self.compute_psi(baseline_preds, current_preds)
        threshold = 0.05
        return DriftResult(
            feature_name="predictions",
            drift_detected=(p_val < threshold) or (psi >= 0.25),
            drift_score=round(max(ks_stat, psi), 6),
            method="ks_test",
            p_value=round(p_val, 6),
            threshold=threshold,
            baseline_stats={
                "mean": float(baseline_preds.mean()),
                "std": float(baseline_preds.std()),
                "psi": psi,
            },
            current_stats={
                "mean": float(current_preds.mean()),
                "std": float(current_preds.std()),
            },
        )

    # ------------------------------------------------------------------
    # Baseline persistence
    # ------------------------------------------------------------------

    def save_baseline(self, df: pd.DataFrame, model_name: str) -> None:
        """Save baseline feature statistics."""
        stats_dict: dict = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats_dict[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "null_count": int(df[col].isna().sum()),
                }
            else:
                stats_dict[col] = {"value_counts": df[col].value_counts().head(50).to_dict()}

        out_path = self.baseline_dir / f"{model_name}_baseline.json"
        out_path.write_text(json.dumps(stats_dict, default=str), encoding="utf-8")
        logger.info("Baseline saved to %s", out_path)

        parquet_path = self.baseline_dir / f"{model_name}_baseline.parquet"
        df.to_parquet(parquet_path, index=False)

    def load_baseline(self, model_name: str) -> Optional[pd.DataFrame]:
        """Load baseline feature statistics."""
        parquet_path = self.baseline_dir / f"{model_name}_baseline.parquet"
        if not parquet_path.exists():
            logger.warning("No baseline found for model '%s' at %s", model_name, parquet_path)
            return None
        return pd.read_parquet(parquet_path)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_drift_report(self, drift_results: list[DriftResult]) -> str:
        """Generate markdown drift report."""
        lines = [
            "# Model Drift Report",
            f"Generated: {datetime.utcnow().isoformat()}",
            "",
            f"**Total features checked:** {len(drift_results)}",
            f"**Features with drift detected:** {sum(1 for d in drift_results if d.drift_detected)}",
            "",
            "| Feature | Drift Detected | Score | Method | P-Value |",
            "|---------|---------------|-------|--------|---------|",
        ]
        for d in sorted(drift_results, key=lambda x: x.drift_score, reverse=True):
            flag = "⚠️ YES" if d.drift_detected else "✅ NO"
            p_str = f"{d.p_value:.4f}" if d.p_value is not None else "N/A"
            lines.append(
                f"| {d.feature_name} | {flag} | {d.drift_score:.4f} | {d.method} | {p_str} |"
            )
        return "\n".join(lines)
