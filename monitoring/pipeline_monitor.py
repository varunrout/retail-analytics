"""
Pipeline health monitoring.
Tracks pipeline run status, latency, and data volume metrics.
"""
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineRun:
    pipeline_name: str
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"  # running / completed / failed
    steps_completed: list[str] = field(default_factory=list)
    steps_failed: list[str] = field(default_factory=list)
    rows_processed: dict = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "pipeline_name": self.pipeline_name,
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "rows_processed": self.rows_processed,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineRun":
        run = cls(
            pipeline_name=d["pipeline_name"],
            run_id=d["run_id"],
            started_at=datetime.fromisoformat(d["started_at"]),
        )
        run.completed_at = datetime.fromisoformat(d["completed_at"]) if d.get("completed_at") else None
        run.status = d.get("status", "unknown")
        run.steps_completed = d.get("steps_completed", [])
        run.steps_failed = d.get("steps_failed", [])
        run.rows_processed = d.get("rows_processed", {})
        run.error_message = d.get("error_message")
        return run


class PipelineMonitor:
    """Tracks and monitors pipeline execution."""

    def __init__(self, log_dir: Path = Path("data/pipeline_logs")) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_run: Optional[PipelineRun] = None

    def _run_log_path(self, pipeline_name: str, run_id: str) -> Path:
        return self.log_dir / f"{pipeline_name}_{run_id}.json"

    def _save_run(self, run: PipelineRun) -> None:
        path = self._run_log_path(run.pipeline_name, run.run_id)
        path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")

    def start_run(self, pipeline_name: str) -> PipelineRun:
        """Start a new pipeline run. Returns run object."""
        run_id = str(uuid.uuid4())[:8]
        self.current_run = PipelineRun(
            pipeline_name=pipeline_name,
            run_id=run_id,
            started_at=datetime.utcnow(),
            status="running",
        )
        self._save_run(self.current_run)
        logger.info("Pipeline '%s' started (run_id=%s)", pipeline_name, run_id)
        return self.current_run

    def log_step(
        self,
        step_name: str,
        rows_processed: int = 0,
        status: str = "completed",
    ) -> None:
        """Log a pipeline step completion."""
        if self.current_run is None:
            logger.warning("No active run; call start_run first.")
            return
        if status == "completed":
            self.current_run.steps_completed.append(step_name)
        else:
            self.current_run.steps_failed.append(step_name)
        if rows_processed > 0:
            self.current_run.rows_processed[step_name] = rows_processed
        self._save_run(self.current_run)
        logger.info(
            "Step '%s' %s (rows=%d)",
            step_name,
            status,
            rows_processed,
        )

    def end_run(
        self,
        status: str = "completed",
        error: Optional[str] = None,
    ) -> PipelineRun:
        """End the current pipeline run."""
        if self.current_run is None:
            raise RuntimeError("No active pipeline run to end.")
        self.current_run.completed_at = datetime.utcnow()
        self.current_run.status = status
        self.current_run.error_message = error
        self._save_run(self.current_run)
        duration = (self.current_run.completed_at - self.current_run.started_at).total_seconds()
        logger.info(
            "Pipeline '%s' %s in %.1fs (run_id=%s)",
            self.current_run.pipeline_name,
            status,
            duration,
            self.current_run.run_id,
        )
        finished = self.current_run
        self.current_run = None
        return finished

    def get_run_history(self, pipeline_name: str, n: int = 10) -> list[PipelineRun]:
        """Get last N runs for a pipeline."""
        log_files = sorted(
            self.log_dir.glob(f"{pipeline_name}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        runs: list[PipelineRun] = []
        for lf in log_files[:n]:
            try:
                data = json.loads(lf.read_text(encoding="utf-8"))
                runs.append(PipelineRun.from_dict(data))
            except Exception as exc:
                logger.warning("Could not parse run log %s: %s", lf, exc)
        return runs

    def check_pipeline_health(self) -> dict:
        """
        Check overall pipeline health.
        Returns: last_run_status, time_since_last_run_hours, avg_latency_minutes,
                 success_rate_7d
        """
        all_runs: list[PipelineRun] = []
        for log_file in self.log_dir.glob("*.json"):
            try:
                data = json.loads(log_file.read_text(encoding="utf-8"))
                all_runs.append(PipelineRun.from_dict(data))
            except Exception:
                pass

        if not all_runs:
            return {
                "last_run_status": "unknown",
                "time_since_last_run_hours": None,
                "avg_latency_minutes": None,
                "success_rate_7d": None,
                "total_runs": 0,
            }

        all_runs.sort(key=lambda r: r.started_at, reverse=True)
        latest = all_runs[0]
        hours_since = (datetime.utcnow() - latest.started_at).total_seconds() / 3600

        completed_runs = [r for r in all_runs if r.completed_at is not None]
        avg_latency: Optional[float] = None
        if completed_runs:
            latencies = [
                (r.completed_at - r.started_at).total_seconds() / 60  # type: ignore[operator]
                for r in completed_runs
            ]
            avg_latency = round(sum(latencies) / len(latencies), 2)

        cutoff = datetime.utcnow() - timedelta(days=7)
        recent = [r for r in all_runs if r.started_at >= cutoff]
        success_rate: Optional[float] = None
        if recent:
            success_rate = round(sum(1 for r in recent if r.status == "completed") / len(recent), 4)

        return {
            "last_run_status": latest.status,
            "time_since_last_run_hours": round(hours_since, 2),
            "avg_latency_minutes": avg_latency,
            "success_rate_7d": success_rate,
            "total_runs": len(all_runs),
        }

    def alert_if_stale(
        self,
        pipeline_name: str,
        max_hours_since_run: int = 26,
    ) -> bool:
        """Return True (and log warning) if pipeline hasn't run recently."""
        history = self.get_run_history(pipeline_name, n=1)
        if not history:
            logger.warning(
                "ALERT: Pipeline '%s' has no recorded runs.", pipeline_name
            )
            return True
        latest = history[0]
        hours_since = (datetime.utcnow() - latest.started_at).total_seconds() / 3600
        if hours_since > max_hours_since_run:
            logger.warning(
                "ALERT: Pipeline '%s' last ran %.1f hours ago (limit %d hours).",
                pipeline_name,
                hours_since,
                max_hours_since_run,
            )
            return True
        return False
