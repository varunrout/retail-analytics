"""
Data quality check definitions for HealthBeauty360.
All checks return a CheckResult with status PASS/WARN/FAIL.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Literal
import logging
import re

logger = logging.getLogger(__name__)

CheckStatus = Literal["PASS", "WARN", "FAIL"]


@dataclass
class CheckResult:
    """Result of a single data quality check."""
    check_name: str
    table_name: str
    status: CheckStatus
    message: str
    rows_checked: int
    rows_failed: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: dict = field(default_factory=dict)


@dataclass
class DQReport:
    """Complete DQ report for a dataset."""
    dataset_name: str
    run_timestamp: datetime
    checks: list[CheckResult]
    overall_status: CheckStatus

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.status == "PASS")

    @property
    def n_warned(self) -> int:
        return sum(1 for c in self.checks if c.status == "WARN")

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if c.status == "FAIL")

    def summary(self) -> str:
        """Return formatted summary string."""
        return (
            f"DQ Report: {self.dataset_name} | "
            f"Overall: {self.overall_status} | "
            f"PASS={self.n_passed} WARN={self.n_warned} FAIL={self.n_failed} | "
            f"Run: {self.run_timestamp.isoformat()}"
        )


class DataQualityChecker:
    """Runs data quality checks on DataFrames."""

    def check_source_freshness(
        self,
        table_name: str,
        last_updated: datetime,
        expected_max_age_hours: int,
    ) -> CheckResult:
        """Check if data is fresh (not older than max age)."""
        age_hours = (datetime.utcnow() - last_updated).total_seconds() / 3600
        if age_hours <= expected_max_age_hours:
            status: CheckStatus = "PASS"
            message = f"Data is fresh ({age_hours:.1f}h old, limit {expected_max_age_hours}h)."
        elif age_hours <= expected_max_age_hours * 1.5:
            status = "WARN"
            message = f"Data is slightly stale ({age_hours:.1f}h old, limit {expected_max_age_hours}h)."
        else:
            status = "FAIL"
            message = f"Data is stale ({age_hours:.1f}h old, limit {expected_max_age_hours}h)."
        return CheckResult(
            check_name="source_freshness",
            table_name=table_name,
            status=status,
            message=message,
            rows_checked=0,
            rows_failed=0,
            details={"age_hours": round(age_hours, 2), "limit_hours": expected_max_age_hours},
        )

    def check_row_count_anomaly(
        self,
        df: pd.DataFrame,
        table_name: str,
        expected_min: int,
        expected_max: int,
    ) -> CheckResult:
        """Check if row count is within expected range."""
        n = len(df)
        if expected_min <= n <= expected_max:
            status: CheckStatus = "PASS"
            message = f"Row count {n} within expected range [{expected_min}, {expected_max}]."
            rows_failed = 0
        else:
            status = "FAIL"
            message = f"Row count {n} outside expected range [{expected_min}, {expected_max}]."
            rows_failed = 1
        return CheckResult(
            check_name="row_count_anomaly",
            table_name=table_name,
            status=status,
            message=message,
            rows_checked=n,
            rows_failed=rows_failed,
            details={"row_count": n, "expected_min": expected_min, "expected_max": expected_max},
        )

    def check_nulls(
        self,
        df: pd.DataFrame,
        table_name: str,
        required_columns: list[str],
    ) -> CheckResult:
        """Check for nulls in required columns. Returns counts per column."""
        null_counts: dict[str, int] = {}
        missing_cols: list[str] = []
        for col in required_columns:
            if col not in df.columns:
                missing_cols.append(col)
            else:
                nc = int(df[col].isna().sum())
                if nc > 0:
                    null_counts[col] = nc

        total_failed = sum(null_counts.values()) + len(missing_cols)
        if missing_cols:
            status: CheckStatus = "FAIL"
            message = f"Missing columns: {missing_cols}. Null counts: {null_counts}."
        elif null_counts:
            status = "FAIL"
            message = f"Null values found: {null_counts}."
        else:
            status = "PASS"
            message = "No nulls in required columns."
        return CheckResult(
            check_name="nulls",
            table_name=table_name,
            status=status,
            message=message,
            rows_checked=len(df),
            rows_failed=total_failed,
            details={"null_counts": null_counts, "missing_columns": missing_cols},
        )

    def check_duplicates(
        self,
        df: pd.DataFrame,
        table_name: str,
        key_columns: list[str],
    ) -> CheckResult:
        """Check for duplicate rows based on key columns."""
        existing_keys = [c for c in key_columns if c in df.columns]
        if not existing_keys:
            return CheckResult(
                check_name="duplicates",
                table_name=table_name,
                status="WARN",
                message=f"Key columns {key_columns} not found in DataFrame.",
                rows_checked=len(df),
                rows_failed=0,
            )
        dupes = int(df.duplicated(subset=existing_keys).sum())
        status: CheckStatus = "PASS" if dupes == 0 else "FAIL"
        message = f"{dupes} duplicate rows found on keys {existing_keys}." if dupes else "No duplicates found."
        return CheckResult(
            check_name="duplicates",
            table_name=table_name,
            status=status,
            message=message,
            rows_checked=len(df),
            rows_failed=dupes,
            details={"duplicate_count": dupes, "key_columns": existing_keys},
        )

    def check_price_sanity(
        self,
        df: pd.DataFrame,
        table_name: str,
        price_col: str,
        min_price: float = 0.01,
        max_price: float = 500.0,
    ) -> CheckResult:
        """Check price values are within realistic range."""
        if price_col not in df.columns:
            return CheckResult(
                check_name="price_sanity",
                table_name=table_name,
                status="WARN",
                message=f"Price column '{price_col}' not found.",
                rows_checked=len(df),
                rows_failed=0,
            )
        prices = pd.to_numeric(df[price_col], errors="coerce")
        out_of_range = int(((prices < min_price) | (prices > max_price) | prices.isna()).sum())
        status: CheckStatus = "PASS" if out_of_range == 0 else ("WARN" if out_of_range / len(df) < 0.01 else "FAIL")
        message = (
            f"{out_of_range} prices outside [{min_price}, {max_price}]."
            if out_of_range
            else f"All prices within [{min_price}, {max_price}]."
        )
        return CheckResult(
            check_name="price_sanity",
            table_name=table_name,
            status=status,
            message=message,
            rows_checked=len(df),
            rows_failed=out_of_range,
            details={"min_price": min_price, "max_price": max_price, "violations": out_of_range},
        )

    def check_gtin_format(
        self,
        df: pd.DataFrame,
        table_name: str,
        gtin_col: str,
    ) -> CheckResult:
        """Validate GTIN/EAN/barcode format (8, 12, or 13 digits)."""
        if gtin_col not in df.columns:
            return CheckResult(
                check_name="gtin_format",
                table_name=table_name,
                status="WARN",
                message=f"GTIN column '{gtin_col}' not found.",
                rows_checked=len(df),
                rows_failed=0,
            )
        pattern = re.compile(r"^\d{8}$|^\d{12}$|^\d{13}$")
        invalid = int(df[gtin_col].astype(str).apply(lambda x: not bool(pattern.match(x))).sum())
        status: CheckStatus = "PASS" if invalid == 0 else ("WARN" if invalid / len(df) < 0.05 else "FAIL")
        message = f"{invalid} invalid GTINs found." if invalid else "All GTINs valid."
        return CheckResult(
            check_name="gtin_format",
            table_name=table_name,
            status=status,
            message=message,
            rows_checked=len(df),
            rows_failed=invalid,
            details={"invalid_count": invalid},
        )

    def check_referential_integrity(
        self,
        fact_df: pd.DataFrame,
        dim_df: pd.DataFrame,
        fact_key: str,
        dim_key: str,
        table_name: str,
    ) -> CheckResult:
        """Check that all FK values in fact table exist in dimension table."""
        if fact_key not in fact_df.columns:
            return CheckResult(
                check_name="referential_integrity",
                table_name=table_name,
                status="FAIL",
                message=f"Key '{fact_key}' missing from fact table.",
                rows_checked=len(fact_df),
                rows_failed=len(fact_df),
            )
        if dim_key not in dim_df.columns:
            return CheckResult(
                check_name="referential_integrity",
                table_name=table_name,
                status="FAIL",
                message=f"Key '{dim_key}' missing from dimension table.",
                rows_checked=len(fact_df),
                rows_failed=len(fact_df),
            )
        dim_keys = set(dim_df[dim_key].dropna().unique())
        orphans = int((~fact_df[fact_key].isin(dim_keys)).sum())
        status: CheckStatus = "PASS" if orphans == 0 else "FAIL"
        message = f"{orphans} orphan keys in '{fact_key}'." if orphans else "Referential integrity OK."
        return CheckResult(
            check_name="referential_integrity",
            table_name=table_name,
            status=status,
            message=message,
            rows_checked=len(fact_df),
            rows_failed=orphans,
            details={"orphan_count": orphans, "fact_key": fact_key, "dim_key": dim_key},
        )

    def check_value_range(
        self,
        df: pd.DataFrame,
        table_name: str,
        col: str,
        min_val: float,
        max_val: float,
    ) -> CheckResult:
        """Generic range check."""
        if col not in df.columns:
            return CheckResult(
                check_name=f"value_range_{col}",
                table_name=table_name,
                status="WARN",
                message=f"Column '{col}' not found.",
                rows_checked=len(df),
                rows_failed=0,
            )
        vals = pd.to_numeric(df[col], errors="coerce")
        out = int(((vals < min_val) | (vals > max_val) | vals.isna()).sum())
        status: CheckStatus = "PASS" if out == 0 else "FAIL"
        message = f"{out} values in '{col}' outside [{min_val}, {max_val}]." if out else f"All '{col}' values in range."
        return CheckResult(
            check_name=f"value_range_{col}",
            table_name=table_name,
            status=status,
            message=message,
            rows_checked=len(df),
            rows_failed=out,
            details={"col": col, "min_val": min_val, "max_val": max_val, "violations": out},
        )

    def check_date_continuity(
        self,
        df: pd.DataFrame,
        table_name: str,
        date_col: str,
        max_gap_days: int = 7,
    ) -> CheckResult:
        """Check for unexpected gaps in date sequence."""
        if date_col not in df.columns:
            return CheckResult(
                check_name="date_continuity",
                table_name=table_name,
                status="WARN",
                message=f"Date column '{date_col}' not found.",
                rows_checked=len(df),
                rows_failed=0,
            )
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values().unique()
        if len(dates) < 2:
            return CheckResult(
                check_name="date_continuity",
                table_name=table_name,
                status="WARN",
                message="Not enough dates to check continuity.",
                rows_checked=len(df),
                rows_failed=0,
            )
        gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
        large_gaps = int((gaps > max_gap_days).sum())
        status: CheckStatus = "PASS" if large_gaps == 0 else "WARN"
        message = (
            f"{large_gaps} date gap(s) exceeding {max_gap_days} days."
            if large_gaps
            else f"No date gaps > {max_gap_days} days found."
        )
        return CheckResult(
            check_name="date_continuity",
            table_name=table_name,
            status=status,
            message=message,
            rows_checked=len(df),
            rows_failed=large_gaps,
            details={"max_gap_days": max_gap_days, "gaps_found": large_gaps},
        )

    def run_all_checks(
        self,
        dataset_name: str,
        df: pd.DataFrame,
        required_cols: Optional[list[str]] = None,
        key_cols: Optional[list[str]] = None,
        price_col: Optional[str] = None,
    ) -> DQReport:
        """Run all applicable checks and return DQReport."""
        results: list[CheckResult] = []
        table_name = dataset_name

        results.append(self.check_row_count_anomaly(df, table_name, expected_min=1, expected_max=10_000_000))

        if required_cols:
            results.append(self.check_nulls(df, table_name, required_cols))

        if key_cols:
            results.append(self.check_duplicates(df, table_name, key_cols))

        if price_col:
            results.append(self.check_price_sanity(df, table_name, price_col))

        date_candidates = [c for c in df.columns if "date" in c.lower() or "dt" in c.lower()]
        if date_candidates:
            results.append(self.check_date_continuity(df, table_name, date_candidates[0]))

        statuses = [r.status for r in results]
        if "FAIL" in statuses:
            overall: CheckStatus = "FAIL"
        elif "WARN" in statuses:
            overall = "WARN"
        else:
            overall = "PASS"

        return DQReport(
            dataset_name=dataset_name,
            run_timestamp=datetime.utcnow(),
            checks=results,
            overall_status=overall,
        )
