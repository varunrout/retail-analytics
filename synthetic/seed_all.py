"""
Orchestrates all synthetic data generators for HealthBeauty360.

Run this script to seed every synthetic dataset used by the platform::

    python -m synthetic.seed_all

or directly::

    python synthetic/seed_all.py
"""

import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER = r"""
  _   _            _ _   _     ____                  _         _____  __  ___
 | | | | ___  __ _| | |_| |__ | __ )  ___  __ _ _  _| |_ _   _|___ / / /_/ _ \
 | |_| |/ _ \/ _` | | __| '_ \|  _ \ / _ \/ _` | | | __| | | | |_ \| '_ \| | |
 |  _  |  __/ (_| | | |_| | | | |_) |  __/ (_| | |_| |_| |_| |___) | (_) | |_|
 |_| |_|\___|\__,_|_|\__|_| |_|____/ \___|\__,_|\__|\__|\__, |____/ \___/ \___/
                                                         |___/
  UK Health & Beauty Retail Intelligence Platform — Synthetic Data Seeder
"""


def print_banner() -> None:
    """Print the HealthBeauty360 ASCII art banner."""
    print(_BANNER)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_output(
    df,
    name: str,
    min_rows: int,
    required_cols: list[str],
) -> bool:
    """Validate a generated DataFrame meets minimum quality expectations.

    Parameters
    ----------
    df:
        DataFrame to validate.
    name:
        Human-readable dataset name used in log messages.
    min_rows:
        Minimum acceptable row count.
    required_cols:
        Column names that must be present and not entirely null.

    Returns
    -------
    bool
        ``True`` if all checks pass, ``False`` otherwise.
    """
    ok = True

    if df is None:
        logger.error("[%s] DataFrame is None.", name)
        return False

    if len(df) < min_rows:
        logger.error(
            "[%s] Too few rows: expected >= %d, got %d.", name, min_rows, len(df)
        )
        ok = False

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error("[%s] Missing columns: %s", name, missing_cols)
        ok = False

    for col in required_cols:
        if col in df.columns and df[col].isna().all():
            logger.error("[%s] Column '%s' is entirely null.", name, col)
            ok = False

    if ok:
        logger.info("[%s] Validation passed (%d rows).", name, len(df))
    return ok


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_all(force_refresh: bool = False) -> dict[str, dict]:
    """Run all synthetic data generators in sequence.

    Steps
    -----
    1. Generate SKU master (shared basis for inventory & costs)
    2. Generate inventory
    3. Generate costs
    4. Generate customers & campaigns
    5. Generate transactions

    Parameters
    ----------
    force_refresh:
        When ``True``, regenerate files even if they already exist.
        When ``False`` (default), skip datasets whose output parquet file
        is already present on disk.

    Returns
    -------
    dict[str, dict]
        Status report keyed by dataset name.  Each value is a dict with
        keys ``status`` (``"ok"`` | ``"skipped"`` | ``"error"``),
        ``rows`` (int), and ``elapsed_s`` (float).
    """
    from synthetic.config import get_synthetic_path  # local import to keep module-level clean

    print_banner()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    results: dict[str, dict] = {}
    overall_start = time.perf_counter()

    def _step(
        label: str,
        emoji: str,
        output_filename: str,
        fn,
        min_rows: int,
        required_cols: list[str],
    ) -> dict:
        """Run a single generation step with timing and validation."""
        out_path = get_synthetic_path(output_filename)
        if not force_refresh and out_path.exists():
            print(f"  {emoji}  [{label}] SKIPPED — {out_path.name} already exists.")
            return {"status": "skipped", "rows": 0, "elapsed_s": 0.0}

        print(f"  {emoji}  [{label}] Generating …", flush=True)
        t0 = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 2)
            logger.exception("[%s] Generation failed: %s", label, exc)
            return {"status": "error", "rows": 0, "elapsed_s": elapsed}

        elapsed = round(time.perf_counter() - t0, 2)

        # result may be a tuple (customers, campaigns)
        df_to_validate = result[0] if isinstance(result, tuple) else result
        valid = validate_output(df_to_validate, label, min_rows, required_cols)
        rows = len(df_to_validate) if df_to_validate is not None else 0
        status = "ok" if valid else "warning"
        print(f"        ✓  {rows:,} rows  ({elapsed}s)")
        return {"status": status, "rows": rows, "elapsed_s": elapsed}

    # ------------------------------------------------------------------
    # Lazy imports (keep top of file clean; these pull in numpy/pandas)
    # ------------------------------------------------------------------
    from synthetic.generate_inventory import generate_sku_master, generate_inventory, save_inventory
    from synthetic.generate_costs import generate_costs, save_costs
    from synthetic.generate_crm import generate_customers, generate_campaigns, save_customers, save_campaigns
    from synthetic.generate_transactions import generate_base_transactions, save_transactions
    from synthetic.config import RANDOM_SEED

    print("\n  Starting synthetic data generation for HealthBeauty360 …\n")

    # Step 1 — Inventory (includes SKU master generation internally)
    def _gen_inventory():
        sku_df = generate_sku_master(seed=RANDOM_SEED)
        inv_df = generate_inventory(sku_df, seed=RANDOM_SEED)
        save_inventory(inv_df)
        return inv_df

    results["inventory"] = _step(
        label="Inventory",
        emoji="📦",
        output_filename="inventory.parquet",
        fn=_gen_inventory,
        min_rows=400,
        required_cols=["sku_id", "stock_on_hand", "safety_stock", "reorder_point"],
    )

    # Step 2 — Costs
    def _gen_costs():
        sku_df = generate_sku_master(seed=RANDOM_SEED)
        cost_df = generate_costs(sku_df, seed=RANDOM_SEED)
        save_costs(cost_df)
        return cost_df

    results["costs"] = _step(
        label="Costs",
        emoji="💰",
        output_filename="costs.parquet",
        fn=_gen_costs,
        min_rows=400,
        required_cols=["sku_id", "landed_cost_gbp", "gross_margin_pct"],
    )

    # Step 3 — Customers
    def _gen_customers():
        cust_df = generate_customers(seed=RANDOM_SEED)
        save_customers(cust_df)
        return cust_df

    results["customers"] = _step(
        label="Customers",
        emoji="👤",
        output_filename="customers.parquet",
        fn=_gen_customers,
        min_rows=8_000,
        required_cols=["customer_id", "customer_lifetime_value_gbp", "loyalty_tier"],
    )

    # Step 4 — Campaigns (depends on customers being generated)
    def _gen_campaigns():
        import pandas as pd
        cust_path = get_synthetic_path("customers.parquet")
        cust_df = pd.read_parquet(cust_path)
        camp_df = generate_campaigns(cust_df, seed=RANDOM_SEED)
        save_campaigns(camp_df)
        return camp_df

    results["campaigns"] = _step(
        label="Campaigns",
        emoji="📧",
        output_filename="campaigns.parquet",
        fn=_gen_campaigns,
        min_rows=5_000,
        required_cols=["campaign_id", "customer_id", "conversion_flag"],
    )

    # Step 5 — Transactions
    def _gen_transactions():
        tx_df = generate_base_transactions(seed=RANDOM_SEED)
        save_transactions(tx_df)
        return tx_df

    results["transactions"] = _step(
        label="Transactions",
        emoji="🛒",
        output_filename="transactions.parquet",
        fn=_gen_transactions,
        min_rows=40_000,
        required_cols=["invoice_no", "stock_code", "net_revenue_gbp", "channel"],
    )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    total_elapsed = round(time.perf_counter() - overall_start, 2)
    print("\n" + "─" * 60)
    print(f"  {'Dataset':<20} {'Status':<10} {'Rows':>10}  {'Time':>8}")
    print("─" * 60)
    for name, info in results.items():
        rows_str = f"{info['rows']:,}" if info["rows"] else "—"
        elapsed_str = f"{info['elapsed_s']}s" if info["elapsed_s"] else "—"
        print(f"  {name:<20} {info['status']:<10} {rows_str:>10}  {elapsed_str:>8}")
    print("─" * 60)
    print(f"  Total elapsed: {total_elapsed}s")

    errors = [k for k, v in results.items() if v["status"] == "error"]
    if errors:
        print(f"\n  ⚠  Errors in: {', '.join(errors)}")
        sys.exit(1)

    print("\n  ✅  All synthetic datasets ready.\n")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed all HealthBeauty360 synthetic data.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate files even if they already exist on disk.",
    )
    args = parser.parse_args()
    run_all(force_refresh=args.force)
