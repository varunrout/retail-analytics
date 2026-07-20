"""The serving layer must return real artifact values, never fabricated numbers."""

import importlib
from pathlib import Path

import pandas as pd
import pytest

from models.demand_forecast import DemandForecastModel
from synthetic.generate_transactions import generate_base_transactions


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch):
    # Train a real demand artifact into a tmp models dir.
    tx = generate_base_transactions(n=6_000)
    DemandForecastModel(artifact_dir=tmp_path).train(tx, forecast_horizon_weeks=4)

    import serving.api as api
    importlib.reload(api)
    monkeypatch.setattr(api, "MODELS_DIR", tmp_path)
    api._CACHE.clear()
    from fastapi.testclient import TestClient

    return TestClient(api.app), tmp_path


def test_forecast_endpoint_serves_artifact_rows(api_client) -> None:
    client, models_dir = api_client
    predictions = pd.read_parquet(models_dir / "demand_forecast_predictions.parquet")
    sku = predictions["stock_code"].iloc[0]

    response = client.get(f"/forecast/{sku}")
    assert response.status_code == 200
    body = response.json()
    expected = float(
        predictions[predictions["stock_code"] == sku].sort_values("forecast_date")["forecast_units"].iloc[0]
    )
    assert abs(body["forecasts"][0]["units_forecast"] - expected) < 1e-6
    assert body["forecasts"][0]["method_used"] != "demo_seasonal"


def test_unknown_ids_return_404(api_client) -> None:
    client, _ = api_client
    assert client.get("/forecast/NOT_A_SKU").status_code == 404


def test_no_fabrication_in_source() -> None:
    source = Path("serving/api.py").read_text()
    assert "np.random" not in source
    assert "random" not in source.replace("# ", "")
