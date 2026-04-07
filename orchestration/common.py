"""Shared orchestration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data_quality.checks import CheckResult, DataQualityChecker
from features.feature_store import FeatureStore
from models.model_utils import save_json


def _checks_payload(checks: list[CheckResult]) -> list[dict[str, Any]]:
    return [
        {
            "check_name": check.check_name,
            "table_name": check.table_name,
            "status": check.status,
            "message": check.message,
            "rows_checked": check.rows_checked,
            "rows_failed": check.rows_failed,
            "details": check.details,
            "timestamp": check.timestamp.isoformat(),
        }
        for check in checks
    ]


def run_standard_data_quality_checks(tables: dict[str, pd.DataFrame]) -> dict[str, list[CheckResult]]:
    checker = DataQualityChecker()
    reports: dict[str, list[CheckResult]] = {}

    if "transactions" in tables:
        transactions = tables["transactions"]
        reports["transactions"] = [
            checker.check_row_count_anomaly(transactions, "transactions", expected_min=1_000, expected_max=5_000_000),
            checker.check_nulls(transactions, "transactions", ["stock_code", "invoice_date", "quantity"]),
            checker.check_duplicates(transactions, "transactions", ["invoice_id", "stock_code", "invoice_date", "quantity", "unit_price_gbp"]),
            checker.check_price_sanity(transactions, "transactions", "unit_price_gbp"),
        ]

    if "inventory" in tables:
        inventory = tables["inventory"]
        reports["inventory"] = [
            checker.check_row_count_anomaly(inventory, "inventory", expected_min=50, expected_max=50_000),
            checker.check_nulls(inventory, "inventory", ["stock_code", "stock_on_hand", "reorder_point"]),
            checker.check_duplicates(inventory, "inventory", ["stock_code"]),
        ]

    if "customers" in tables:
        customers = tables["customers"]
        reports["customers"] = [
            checker.check_row_count_anomaly(customers, "customers", expected_min=100, expected_max=2_000_000),
            checker.check_nulls(customers, "customers", ["customer_id", "loyalty_tier", "customer_lifetime_value_gbp"]),
            checker.check_duplicates(customers, "customers", ["customer_id"]),
        ]
    return reports


def write_data_quality_report(reports: dict[str, list[CheckResult]], report_path: Path) -> Path:
    payload = {
        table_name: _checks_payload(checks)
        for table_name, checks in reports.items()
    }
    return save_json(payload, report_path)


def build_feature_outputs(
    tables: dict[str, pd.DataFrame],
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    store = FeatureStore(data_dir=data_dir)
    return store.run(
        transactions_df=tables.get("transactions"),
        inventory_df=tables.get("inventory"),
        costs_df=tables.get("costs"),
        crm_df=tables.get("customers"),
    )


def overall_status(reports: dict[str, list[CheckResult]]) -> str:
    flattened = [check.status for checks in reports.values() for check in checks]
    if any(status == "FAIL" for status in flattened):
        return "FAIL"
    if any(status == "WARN" for status in flattened):
        return "WARN"
    return "PASS"