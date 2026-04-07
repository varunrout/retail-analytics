"""Daily orchestration pipeline for operational refreshes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from models.churn_prediction import ChurnPredictionModel
from models.inventory_scoring import InventoryScoringModel
from models.model_utils import DATA_DIR, MODELS_DIR, REPORTS_DIR, load_input_tables, save_json
from models.trend_detection import TrendDetectionModel
from monitoring.pipeline_monitor import PipelineMonitor
from orchestration.common import build_feature_outputs, overall_status, run_standard_data_quality_checks, write_data_quality_report


def run_daily_pipeline(
    force_refresh: bool = False,
    tables: dict[str, pd.DataFrame] | None = None,
    data_dir: Path | None = None,
    artifact_dir: Path | None = None,
    report_dir: Path | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    runtime_data_dir = data_dir or DATA_DIR
    runtime_artifact_dir = artifact_dir or MODELS_DIR
    runtime_report_dir = report_dir or REPORTS_DIR
    runtime_log_dir = log_dir or (runtime_data_dir / "pipeline_logs")
    runtime_data_dir.mkdir(parents=True, exist_ok=True)
    runtime_artifact_dir.mkdir(parents=True, exist_ok=True)
    runtime_report_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_dir.mkdir(parents=True, exist_ok=True)

    monitor = PipelineMonitor(log_dir=runtime_log_dir)
    monitor.start_run("daily_pipeline")
    try:
        pipeline_tables = tables or load_input_tables(force_refresh=force_refresh)
        monitor.log_step("load_inputs", rows_processed=len(pipeline_tables.get("transactions", [])))

        dq_reports = run_standard_data_quality_checks(pipeline_tables)
        dq_report_path = write_data_quality_report(dq_reports, runtime_report_dir / "daily_data_quality.json")
        monitor.log_step("data_quality", rows_processed=sum(len(checks) for checks in dq_reports.values()))

        feature_outputs = build_feature_outputs(pipeline_tables, data_dir=runtime_data_dir)
        monitor.log_step("feature_store", rows_processed=sum(len(frame) for frame in feature_outputs.values()))

        inventory_result = InventoryScoringModel(artifact_dir=runtime_artifact_dir).train(
            inventory_df=pipeline_tables["inventory"],
            transactions_df=pipeline_tables["transactions"],
            costs_df=pipeline_tables.get("costs"),
        )
        churn_result = ChurnPredictionModel(artifact_dir=runtime_artifact_dir).train(
            transactions_df=pipeline_tables["transactions"],
            crm_df=pipeline_tables.get("customers"),
            campaigns_df=pipeline_tables.get("campaigns"),
        )
        trend_result = TrendDetectionModel(artifact_dir=runtime_artifact_dir).run(pipeline_tables["transactions"])
        monitor.log_step(
            "daily_models",
            rows_processed=(
                len(inventory_result["scores"]) + len(churn_result["scores"]) + len(trend_result["sku_trends"])
            ),
        )

        summary = {
            "pipeline_name": "daily_pipeline",
            "status": "completed",
            "data_quality_status": overall_status(dq_reports),
            "data_quality_report": str(dq_report_path),
            "feature_outputs": {name: list(frame.shape) for name, frame in feature_outputs.items()},
            "model_outputs": {
                "inventory_scoring_rows": len(inventory_result["scores"]),
                "churn_scores_rows": len(churn_result["scores"]),
                "trend_detection_rows": len(trend_result["sku_trends"]),
            },
        }
        summary_path = save_json(summary, runtime_report_dir / "daily_pipeline_summary.json")
        monitor.log_step("write_summary", rows_processed=1)
        monitor.end_run(status="completed")
        summary["summary_path"] = str(summary_path)
        return summary
    except Exception as exc:
        monitor.end_run(status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    result = run_daily_pipeline()
    print(result)