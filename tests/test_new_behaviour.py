"""Tests for the honesty/rigour changes: baselines, decisions, new models."""

from pathlib import Path

from models.customer_segmentation import CustomerSegmentationModel
from models.demand_forecast import DemandForecastModel
from models.price_elasticity import PriceElasticityModel
from models.trend_detection import TrendDetectionModel
from synthetic.generate_crm import generate_customers
from synthetic.generate_inventory import generate_inventory, generate_sku_master
from synthetic.generate_transactions import generate_base_transactions


def _tables(n_skus: int = 60, n_tx: int = 8_000) -> dict:
    sku_df = generate_sku_master(n_skus=n_skus)
    return {
        "transactions": generate_base_transactions(n=n_tx),
        "inventory": generate_inventory(sku_df),
        "customers": generate_customers(n=400),
    }


def test_demand_forecast_reports_baseline_and_reorder(tmp_path: Path) -> None:
    tables = _tables()
    result = DemandForecastModel(artifact_dir=tmp_path).train(tables["transactions"], forecast_horizon_weeks=4)
    metrics = result["metrics"]
    forecasts = result["forecasts"]
    # Baseline comparison exists and is a real number, not silently zero.
    assert {"baseline_mape", "beats_baseline", "reorder_point"}.issubset(metrics.columns)
    evaluated = metrics[metrics["method_used"] == "ridge_recursive"]
    assert evaluated["baseline_mape"].mean() > 0
    # Every forecast ends in a decision.
    assert "reorder_point" in forecasts.columns
    assert (forecasts["reorder_point"] >= 0).all()
    summary = result["summary"]
    assert 0.0 <= summary.share_beating_baseline <= 1.0


def test_price_elasticity_recovers_negative_elastic_categories(tmp_path: Path) -> None:
    tables = _tables(n_tx=12_000)
    result = PriceElasticityModel(artifact_dir=tmp_path).train(tables["transactions"], inventory_df=tables["inventory"])
    table = result["table"]
    assert {"elasticity", "ci_low", "ci_high", "recommended_max_discount", "recommended_action"}.issubset(table.columns)
    # Elasticities should be negative (higher price -> fewer units).
    assert (table["elasticity"] < 0).mean() >= 0.8
    # Confidence interval is ordered.
    assert (table["ci_low"] <= table["ci_high"]).all()


def test_segmentation_selects_k_by_silhouette(tmp_path: Path) -> None:
    tables = _tables()
    result = CustomerSegmentationModel(artifact_dir=tmp_path).train(tables["transactions"], crm_df=tables["customers"])
    summary = result["summary"]
    assert 2 <= summary.selected_k <= 8
    assert summary.silhouette_by_k  # curve persisted
    # Selected k maximises the recorded silhouette curve.
    best_k = max(summary.silhouette_by_k, key=summary.silhouette_by_k.get)
    assert summary.selected_k == best_k
    assert "recommended_action" in result["segments"].columns


def test_trend_detector_reports_mann_kendall(tmp_path: Path) -> None:
    tables = _tables()
    result = TrendDetectionModel(artifact_dir=tmp_path).run(tables["transactions"])
    sku = result["sku_trends"]
    assert {"mk_trend", "mk_pvalue", "mk_significant", "action_rank"}.issubset(sku.columns)
    assert sku["mk_pvalue"].between(0, 1).all()
    # Significant accelerating/declining labels require a significant MK test.
    labelled = sku[sku["trend_label"].isin(["accelerating", "declining"])]
    assert labelled.empty or labelled["mk_significant"].all()
