"""
Synthetic CRM data generator.

Generates 10,000 customers with purchase history and loyalty metadata,
plus a campaign contact history covering ~20 email/SMS/push campaigns
over 2022–2024.

Writes:
  - ``data/synthetic/customers.parquet``
  - ``data/synthetic/campaigns.parquet``
"""

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from synthetic.config import (
    N_CUSTOMERS,
    RANDOM_SEED,
    UK_CATEGORIES,
    get_synthetic_path,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UK postcode area prefixes (ONS NUTS-level approximation)
# ---------------------------------------------------------------------------
_UK_POSTCODE_AREAS: list[str] = [
    "SW1", "SW2", "SE1", "SE5", "E1", "E2", "EC1", "WC1", "W1", "N1",
    "NW1", "M1", "M2", "B1", "B2", "LS1", "LS2", "BS1", "BS2", "CF1",
    "EH1", "EH2", "G1", "G2", "L1", "L2", "OX1", "CB1", "MK1", "RG1",
    "SO1", "BN1", "CT1", "ME1", "TN1", "PE1", "LE1", "DE1", "NG1", "S1",
]

_AGE_BANDS: list[str] = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
_AGE_WEIGHTS: list[float] = [0.10, 0.22, 0.25, 0.20, 0.13, 0.10]

_GENDERS: list[str] = ["F", "M", "Non-binary", "Prefer not to say"]
_GENDER_WEIGHTS: list[float] = [0.72, 0.22, 0.04, 0.02]

_CHANNELS: list[str] = [
    "organic", "paid_search", "social_media", "email", "referral", "marketplace"
]
_CHANNEL_WEIGHTS: list[float] = [0.28, 0.22, 0.20, 0.12, 0.10, 0.08]

_LOYALTY_TIERS: list[str] = ["Bronze", "Silver", "Gold", "Platinum"]


def _loyalty_tier(clv: float) -> str:
    """Map customer lifetime value to a loyalty tier."""
    if clv < 50:
        return "Bronze"
    if clv < 150:
        return "Silver"
    if clv < 350:
        return "Gold"
    return "Platinum"


def generate_customers(n: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic customer records.

    Parameters
    ----------
    n:
        Number of customer records to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``customer_id``, ``email_hash``, ``postcode_area``,
        ``age_band``, ``gender``, ``acquisition_channel``,
        ``first_purchase_date``, ``last_purchase_date``,
        ``customer_lifetime_value_gbp``, ``total_orders``,
        ``is_active``, ``preferred_category``, ``loyalty_tier``.
    """
    rng = np.random.default_rng(seed)
    fake = Faker("en_GB")
    fake.seed_instance(seed)

    customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, n + 1)]

    # Email hashes (SHA-256 of fake email, no PII stored)
    emails = [fake.email() for _ in range(n)]
    email_hashes = [hashlib.sha256(e.encode()).hexdigest() for e in emails]

    postcode_areas = rng.choice(_UK_POSTCODE_AREAS, size=n, replace=True)
    age_bands = rng.choice(_AGE_BANDS, size=n, replace=True, p=_AGE_WEIGHTS)
    genders = rng.choice(_GENDERS, size=n, replace=True, p=_GENDER_WEIGHTS)
    channels = rng.choice(_CHANNELS, size=n, replace=True, p=_CHANNEL_WEIGHTS)
    preferred_categories = rng.choice(UK_CATEGORIES, size=n, replace=True)

    # First purchase dates: 2018-01-01 – 2024-06-30
    origin = pd.Timestamp("2018-01-01")
    end = pd.Timestamp("2024-06-30")
    span_days = (end - origin).days
    first_purchase_offsets = rng.integers(0, span_days, size=n)
    first_purchase_dates = [origin + pd.Timedelta(days=int(d)) for d in first_purchase_offsets]

    # Last purchase: between first_purchase + 1 day and 2024-08-31
    last_end = pd.Timestamp("2024-08-31")
    last_purchase_dates = []
    for fp in first_purchase_dates:
        remaining = max((last_end - fp).days, 1)
        offset = rng.integers(1, remaining + 1)
        last_purchase_dates.append(fp + pd.Timedelta(days=int(offset)))

    # CLV: log-normal (median ~£85, long tail up to ~£2 000)
    # log-normal params: mu = ln(85), sigma = 0.9
    mu_ln = np.log(85)
    sigma_ln = 0.9
    clv = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n).round(2)

    # is_active: purchased in last 180 days from reference date 2024-09-01
    reference = pd.Timestamp("2024-09-01")
    is_active = [(reference - lp).days <= 180 for lp in last_purchase_dates]

    # Total orders: correlated loosely with CLV
    avg_order_value = rng.uniform(20, 60, size=n)
    total_orders = np.maximum(1, (clv / avg_order_value).astype(int))

    loyalty_tiers = [_loyalty_tier(c) for c in clv]

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "email_hash": email_hashes,
            "postcode_area": postcode_areas,
            "age_band": age_bands,
            "gender": genders,
            "acquisition_channel": channels,
            "first_purchase_date": pd.to_datetime(first_purchase_dates),
            "last_purchase_date": pd.to_datetime(last_purchase_dates),
            "customer_lifetime_value_gbp": clv,
            "total_orders": total_orders,
            "is_active": is_active,
            "preferred_category": preferred_categories,
            "loyalty_tier": loyalty_tiers,
        }
    )
    logger.info(
        "Generated %d customers | active=%d | avg CLV=£%.2f",
        len(df),
        df["is_active"].sum(),
        df["customer_lifetime_value_gbp"].mean(),
    )
    return df


# ---------------------------------------------------------------------------
# Campaign definitions
# ---------------------------------------------------------------------------
_CAMPAIGNS: list[dict] = [
    {"name": "Jan New Year Refresh", "type": "email", "date": "2022-01-10", "segment": "active"},
    {"name": "Valentine's Fragrance Push", "type": "email", "date": "2022-02-07", "segment": "all"},
    {"name": "Mother's Day Skincare", "type": "email", "date": "2022-03-14", "segment": "all"},
    {"name": "Spring Haircare Sale", "type": "sms", "date": "2022-04-04", "segment": "haircare"},
    {"name": "Summer Glow SPF Launch", "type": "email", "date": "2022-06-06", "segment": "sun_care"},
    {"name": "Mid-Year Vitamins Promo", "type": "push", "date": "2022-07-11", "segment": "vitamins"},
    {"name": "Back to School Bath", "type": "email", "date": "2022-09-05", "segment": "all"},
    {"name": "Black Friday Beauty", "type": "email", "date": "2022-11-21", "segment": "all"},
    {"name": "Christmas Gift Sets", "type": "email", "date": "2022-12-05", "segment": "Gold+"},
    {"name": "Blue Monday Wellness", "type": "sms", "date": "2023-01-16", "segment": "active"},
    {"name": "Spring Skincare Edit", "type": "email", "date": "2023-03-20", "segment": "skincare"},
    {"name": "Father's Day Grooming", "type": "email", "date": "2023-06-12", "segment": "all"},
    {"name": "Summer Makeup Masterclass", "type": "push", "date": "2023-07-03", "segment": "makeup"},
    {"name": "Autumn Repair Haircare", "type": "email", "date": "2023-09-11", "segment": "haircare"},
    {"name": "World Mental Health Day", "type": "email", "date": "2023-10-09", "segment": "vitamins"},
    {"name": "Black Friday Mega Sale", "type": "email", "date": "2023-11-20", "segment": "all"},
    {"name": "Christmas Countdown", "type": "sms", "date": "2023-12-04", "segment": "Gold+"},
    {"name": "Dry January Detox", "type": "email", "date": "2024-01-08", "segment": "active"},
    {"name": "Spring Glow-Up Campaign", "type": "email", "date": "2024-03-11", "segment": "skincare"},
    {"name": "Summer Festival Beauty", "type": "push", "date": "2024-06-17", "segment": "makeup"},
]


def generate_campaigns(
    customers_df: pd.DataFrame, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate email/SMS/push campaign contact history.

    Each of the ~20 campaigns targets a subset of customers.  Open, click
    and conversion flags are generated with realistic UK health & beauty
    benchmarks.

    Parameters
    ----------
    customers_df:
        Customer records from :func:`generate_customers`.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per (campaign × customer) contact with engagement flags
        and attributed revenue.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for idx, camp in enumerate(_CAMPAIGNS):
        campaign_id = f"CAMP{str(idx + 1).zfill(3)}"
        sent_date = pd.Timestamp(camp["date"])
        segment = camp["segment"]
        ctype = camp["type"]

        # Filter customers by segment
        if segment == "all":
            targets = customers_df
        elif segment == "active":
            targets = customers_df[customers_df["is_active"]]
        elif segment == "Gold+":
            targets = customers_df[customers_df["loyalty_tier"].isin(["Gold", "Platinum"])]
        else:
            # segment is a category name
            targets = customers_df[customers_df["preferred_category"] == segment]

        if len(targets) == 0:
            targets = customers_df

        # Sample 30–70 % of matching segment
        sample_frac = rng.uniform(0.30, 0.70)
        n_send = max(1, int(len(targets) * sample_frac))
        sampled = targets.sample(n=n_send, random_state=int(rng.integers(0, 99999)))

        # Open / click / conversion rates vary by type
        if ctype == "email":
            open_rate = rng.uniform(0.22, 0.35)
            click_rate = rng.uniform(0.03, 0.08)
            conv_rate = rng.uniform(0.01, 0.03)
        elif ctype == "sms":
            open_rate = rng.uniform(0.55, 0.75)
            click_rate = rng.uniform(0.05, 0.12)
            conv_rate = rng.uniform(0.02, 0.05)
        else:  # push
            open_rate = rng.uniform(0.15, 0.28)
            click_rate = rng.uniform(0.02, 0.06)
            conv_rate = rng.uniform(0.005, 0.02)

        n_s = len(sampled)
        open_flags = rng.random(n_s) < open_rate
        click_flags = open_flags & (rng.random(n_s) < click_rate)
        conv_flags = click_flags & (rng.random(n_s) < conv_rate)

        # Attributed revenue: only for converters, ~£25–£90 per order
        revenue = np.where(conv_flags, rng.uniform(25, 90, size=n_s), 0.0).round(2)

        for i, (_, cust_row) in enumerate(sampled.iterrows()):
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "campaign_name": camp["name"],
                    "campaign_type": ctype,
                    "sent_date": sent_date,
                    "target_segment": segment,
                    "customer_id": cust_row["customer_id"],
                    "open_flag": bool(open_flags[i]),
                    "click_flag": bool(click_flags[i]),
                    "conversion_flag": bool(conv_flags[i]),
                    "revenue_attributed_gbp": float(revenue[i]),
                }
            )

    df = pd.DataFrame(rows)
    df["sent_date"] = pd.to_datetime(df["sent_date"])
    logger.info(
        "Generated %d campaign contact rows across %d campaigns | conversions=%d",
        len(df),
        df["campaign_id"].nunique(),
        df["conversion_flag"].sum(),
    )
    return df


def save_customers(df: pd.DataFrame) -> Path:
    """Save customers DataFrame to ``data/synthetic/customers.parquet``.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`generate_customers`.

    Returns
    -------
    Path
        Absolute path of the written parquet file.
    """
    out_path = get_synthetic_path("customers.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Saved customers to %s (%d rows).", out_path, len(df))
    return out_path


def save_campaigns(df: pd.DataFrame) -> Path:
    """Save campaigns DataFrame to ``data/synthetic/campaigns.parquet``.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`generate_campaigns`.

    Returns
    -------
    Path
        Absolute path of the written parquet file.
    """
    out_path = get_synthetic_path("campaigns.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Saved campaigns to %s (%d rows).", out_path, len(df))
    return out_path


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrate CRM generation and persist both datasets to disk.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(customers_df, campaigns_df)``
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cust_df = generate_customers(seed=RANDOM_SEED)
    camp_df = generate_campaigns(cust_df, seed=RANDOM_SEED)
    save_customers(cust_df)
    save_campaigns(camp_df)
    return cust_df, camp_df


if __name__ == "__main__":
    customers, campaigns = run()
    print(customers.head())
    print(campaigns.head())
