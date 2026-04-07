from pathlib import Path

from models.churn_prediction import ChurnPredictionModel
from models.customer_segmentation import CustomerSegmentationModel
from models.demand_forecast import DemandForecastModel
from models.inventory_scoring import InventoryScoringModel
from models.trend_detection import TrendDetectionModel
from synthetic.generate_costs import generate_costs
from synthetic.generate_crm import generate_campaigns, generate_customers
from synthetic.generate_inventory import generate_inventory, generate_sku_master
from synthetic.generate_transactions import generate_base_transactions


def build_test_tables() -> dict:
    sku_df = generate_sku_master(n_skus=40)
    inventory_df = generate_inventory(sku_df)
    costs_df = generate_costs(sku_df)
    customers_df = generate_customers(n=250)
    campaigns_df = generate_campaigns(customers_df)
    transactions_df = generate_base_transactions(n=2_500)
    return {
        "transactions": transactions_df,
        "inventory": inventory_df,
        "costs": costs_df,
        "customers": customers_df,
        "campaigns": campaigns_df,
    }


def test_demand_forecast_produces_predictions(tmp_path: Path) -> None:
    tables = build_test_tables()
    result = DemandForecastModel(artifact_dir=tmp_path).train(tables["transactions"], forecast_horizon_weeks=4)
    assert not result["forecasts"].empty
    assert result["summary"].sku_count > 0


def test_customer_and_churn_models_produce_customer_outputs(tmp_path: Path) -> None:
    tables = build_test_tables()
    segmentation = CustomerSegmentationModel(artifact_dir=tmp_path).train(tables["transactions"], crm_df=tables["customers"])
    churn = ChurnPredictionModel(artifact_dir=tmp_path).train(
        transactions_df=tables["transactions"],
        crm_df=tables["customers"],
        campaigns_df=tables["campaigns"],
    )
    assert "segment_name" in segmentation["segments"].columns
    assert churn["scores"]["churn_probability"].between(0, 1).all()


def test_inventory_and_trend_models_produce_scores(tmp_path: Path) -> None:
    tables = build_test_tables()
    inventory = InventoryScoringModel(artifact_dir=tmp_path).train(
        inventory_df=tables["inventory"],
        transactions_df=tables["transactions"],
        costs_df=tables["costs"],
    )
    trends = TrendDetectionModel(artifact_dir=tmp_path).run(tables["transactions"])
    assert inventory["scores"]["stockout_risk_score"].ge(0).all()
    assert trends["sku_trends"]["trend_label"].isin(["accelerating", "growing", "declining", "stable"]).all()


def test_transactions_do_not_repeat_sku_within_invoice() -> None:
    transactions = generate_base_transactions(n=2_500)
    duplicate_count = transactions.duplicated(subset=["invoice_no", "stock_code"]).sum()
    assert duplicate_count == 0