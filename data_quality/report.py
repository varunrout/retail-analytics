"""
DQ report generation: creates HTML/markdown summary of data quality checks.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

from data_quality.checks import DQReport, CheckResult, DataQualityChecker

logger = logging.getLogger(__name__)

_STATUS_COLOURS = {"PASS": "#28a745", "WARN": "#ffc107", "FAIL": "#dc3545"}


def generate_markdown_report(report: DQReport) -> str:
    """Generate a markdown-formatted DQ report."""
    lines = [
        f"# Data Quality Report: {report.dataset_name}",
        f"**Run:** {report.run_timestamp.isoformat()}  ",
        f"**Overall Status:** {report.overall_status}  ",
        f"**Checks:** {report.n_passed} PASS / {report.n_warned} WARN / {report.n_failed} FAIL",
        "",
        "## Check Results",
        "",
        "| Check | Table | Status | Rows Checked | Rows Failed | Message |",
        "|-------|-------|--------|-------------|-------------|---------|",
    ]
    for c in report.checks:
        lines.append(
            f"| {c.check_name} | {c.table_name} | **{c.status}** "
            f"| {c.rows_checked:,} | {c.rows_failed:,} | {c.message} |"
        )
    failed = [c for c in report.checks if c.status == "FAIL"]
    if failed:
        lines += ["", "## Failed Check Details", ""]
        for c in failed:
            lines.append(f"### {c.check_name} ({c.table_name})")
            lines.append(f"- **Message:** {c.message}")
            if c.details:
                lines.append(f"- **Details:** `{c.details}`")
            lines.append("")
    return "\n".join(lines)


def generate_html_report(report: DQReport) -> str:
    """Generate HTML report with color-coded status badges."""

    def badge(status: str) -> str:
        colour = _STATUS_COLOURS.get(status, "#6c757d")
        return (
            f'<span style="background:{colour};color:#fff;padding:2px 8px;'
            f'border-radius:4px;font-weight:bold;">{status}</span>'
        )

    rows = ""
    for c in report.checks:
        rows += (
            f"<tr><td>{c.check_name}</td><td>{c.table_name}</td>"
            f"<td>{badge(c.status)}</td><td>{c.rows_checked:,}</td>"
            f"<td>{c.rows_failed:,}</td><td>{c.message}</td></tr>\n"
        )

    overall_colour = _STATUS_COLOURS.get(report.overall_status, "#6c757d")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>DQ Report: {report.dataset_name}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 24px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; }}
  th {{ background: #343a40; color: #fff; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
</style></head>
<body>
<h1>Data Quality Report: {report.dataset_name}</h1>
<p><strong>Run:</strong> {report.run_timestamp.isoformat()}</p>
<p><strong>Overall Status:</strong> {badge(report.overall_status)}</p>
<p>
  <strong>Checks:</strong>
  {report.n_passed} PASS /
  {report.n_warned} WARN /
  {report.n_failed} FAIL
</p>
<table>
<thead><tr>
  <th>Check</th><th>Table</th><th>Status</th>
  <th>Rows Checked</th><th>Rows Failed</th><th>Message</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>"""
    return html


def save_report(
    report: DQReport,
    output_dir: Path = Path("data/reports"),
) -> tuple[Path, Path]:
    """Save both markdown and HTML reports. Returns (md_path, html_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = report.run_timestamp.strftime("%Y%m%dT%H%M%S")
    safe_name = report.dataset_name.replace(" ", "_").lower()
    md_path = output_dir / f"dq_{safe_name}_{ts}.md"
    html_path = output_dir / f"dq_{safe_name}_{ts}.html"

    md_path.write_text(generate_markdown_report(report), encoding="utf-8")
    html_path.write_text(generate_html_report(report), encoding="utf-8")
    logger.info("DQ report saved: %s, %s", md_path, html_path)
    return md_path, html_path


def run_full_dq_suite(data_dir: Path = Path("data")) -> DQReport:
    """
    Run DQ checks against all available datasets.
    Loads parquet files from data/synthetic/ and data/bronze/.
    Returns combined DQReport.
    """
    checker = DataQualityChecker()
    all_results: list = []

    search_dirs = [data_dir / "synthetic", data_dir / "bronze"]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for parquet_file in search_dir.rglob("*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                dataset_name = parquet_file.stem
                sub_report = checker.run_all_checks(dataset_name, df)
                all_results.extend(sub_report.checks)
                logger.info("DQ checked: %s (%d rows)", dataset_name, len(df))
            except Exception as exc:
                logger.warning("Failed to DQ check %s: %s", parquet_file, exc)

    if not all_results:
        from data_quality.checks import CheckResult
        all_results = [
            CheckResult(
                check_name="no_data",
                table_name="none",
                status="WARN",
                message="No parquet files found for DQ checks.",
                rows_checked=0,
                rows_failed=0,
            )
        ]

    statuses = [r.status for r in all_results]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "WARN"
    else:
        overall = "PASS"

    return DQReport(
        dataset_name="full_suite",
        run_timestamp=datetime.utcnow(),
        checks=all_results,
        overall_status=overall,
    )
