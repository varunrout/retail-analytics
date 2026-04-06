"""
Source freshness monitoring.
Tracks last-updated timestamps for all data sources and alerts on staleness.
"""
import os
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FreshnessCheck:
    source_name: str
    last_updated: Optional[datetime]
    expected_max_age_hours: int
    status: str  # FRESH / STALE / MISSING
    hours_since_update: Optional[float]


class FreshnessMonitor:
    """Monitors data source freshness."""

    SOURCES: dict[str, dict] = {
        "bank_holidays": {
            "max_age_hours": 24 * 30,
            "path": "data/raw/bank_holidays/bank_holidays.json",
        },
        "weather": {"max_age_hours": 25, "path": "data/raw/weather/"},
        "ons_retail_sales": {
            "max_age_hours": 24 * 7,
            "path": "data/raw/ons_retail_sales/",
        },
        "google_trends": {
            "max_age_hours": 24 * 7,
            "path": "data/raw/google_trends/",
        },
        "uci_transactions": {
            "max_age_hours": 24 * 365,
            "path": "data/raw/uci_transactions/",
        },
        "shopify_products": {
            "max_age_hours": 25,
            "path": "data/raw/shopify_products.parquet",
        },
        "ebay_listings": {
            "max_age_hours": 25,
            "path": "data/raw/ebay_listings.parquet",
        },
    }

    def _get_path_mtime(self, path_str: str) -> Optional[datetime]:
        """Return last-modified time for a file or the most recent file in a dir."""
        p = Path(path_str)
        if not p.exists():
            return None
        if p.is_file():
            return datetime.utcfromtimestamp(p.stat().st_mtime)
        # Directory: pick the most recently modified file
        files = list(p.rglob("*"))
        files = [f for f in files if f.is_file()]
        if not files:
            return None
        latest = max(files, key=lambda f: f.stat().st_mtime)
        return datetime.utcfromtimestamp(latest.stat().st_mtime)

    def check_source(self, source_name: str) -> FreshnessCheck:
        """Check freshness of a single source."""
        config = self.SOURCES.get(source_name)
        if config is None:
            return FreshnessCheck(
                source_name=source_name,
                last_updated=None,
                expected_max_age_hours=0,
                status="MISSING",
                hours_since_update=None,
            )

        max_age = config["max_age_hours"]
        last_updated = self._get_path_mtime(config["path"])

        if last_updated is None:
            return FreshnessCheck(
                source_name=source_name,
                last_updated=None,
                expected_max_age_hours=max_age,
                status="MISSING",
                hours_since_update=None,
            )

        hours_since = (datetime.utcnow() - last_updated).total_seconds() / 3600
        status = "FRESH" if hours_since <= max_age else "STALE"
        return FreshnessCheck(
            source_name=source_name,
            last_updated=last_updated,
            expected_max_age_hours=max_age,
            status=status,
            hours_since_update=round(hours_since, 2),
        )

    def check_all_sources(self) -> list[FreshnessCheck]:
        """Check all configured sources."""
        return [self.check_source(name) for name in self.SOURCES]

    def get_stale_sources(self) -> list[FreshnessCheck]:
        """Return only stale or missing sources."""
        return [c for c in self.check_all_sources() if c.status in ("STALE", "MISSING")]

    def report(self) -> str:
        """Return formatted freshness report."""
        checks = self.check_all_sources()
        lines = ["# Freshness Report", f"Generated: {datetime.utcnow().isoformat()}", ""]
        lines.append(f"{'Source':<30} {'Status':<10} {'Hours Since Update':<22} {'Max Age (h)'}")
        lines.append("-" * 80)
        for c in checks:
            hours_str = f"{c.hours_since_update:.1f}" if c.hours_since_update is not None else "N/A"
            lines.append(f"{c.source_name:<30} {c.status:<10} {hours_str:<22} {c.expected_max_age_hours}")
        stale = [c for c in checks if c.status in ("STALE", "MISSING")]
        lines.append("")
        lines.append(f"Summary: {len(checks)} sources checked, {len(stale)} stale/missing.")
        return "\n".join(lines)
