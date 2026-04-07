from pathlib import Path

from orchestration.daily_pipeline import run_daily_pipeline
from orchestration.weekly_pipeline import run_weekly_pipeline
from tests.test_models import build_test_tables


def test_daily_pipeline_writes_summary(tmp_path: Path) -> None:
    tables = build_test_tables()
    result = run_daily_pipeline(
        tables=tables,
        data_dir=tmp_path / "data",
        artifact_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        log_dir=tmp_path / "logs",
    )
    assert result["status"] == "completed"
    assert (tmp_path / "reports" / "daily_pipeline_summary.json").exists()


def test_weekly_pipeline_writes_summary(tmp_path: Path) -> None:
    tables = build_test_tables()
    result = run_weekly_pipeline(
        tables=tables,
        data_dir=tmp_path / "data",
        artifact_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        log_dir=tmp_path / "logs",
        baseline_dir=tmp_path / "baselines",
    )
    assert result["status"] == "completed"
    assert (tmp_path / "reports" / "weekly_pipeline_summary.json").exists()