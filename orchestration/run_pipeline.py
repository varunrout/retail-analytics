"""CLI wrapper for daily and weekly pipelines."""

from __future__ import annotations

import argparse
import json

from orchestration.daily_pipeline import run_daily_pipeline
from orchestration.weekly_pipeline import run_weekly_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HealthBeauty360 orchestration pipelines.")
    parser.add_argument("--daily", action="store_true", help="Run the daily operational pipeline.")
    parser.add_argument("--weekly", action="store_true", help="Run the weekly retraining pipeline.")
    parser.add_argument("--full", action="store_true", help="Run daily then weekly pipelines.")
    parser.add_argument("--force-refresh", action="store_true", help="Regenerate synthetic input data before running.")
    args = parser.parse_args()

    if not any([args.daily, args.weekly, args.full]):
        parser.error("Select at least one of --daily, --weekly, or --full.")

    results: dict[str, object] = {}
    if args.daily or args.full:
        results["daily"] = run_daily_pipeline(force_refresh=args.force_refresh)
    if args.weekly or args.full:
        results["weekly"] = run_weekly_pipeline(force_refresh=args.force_refresh)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()