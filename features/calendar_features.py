"""
Calendar and temporal features.
Computes seasonality indices, holiday effects, and time-based features.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def add_calendar_features(df: pd.DataFrame, date_col: str = "invoice_date") -> pd.DataFrame:
    """
    Add time-based features to a DataFrame with a date column.

    Adds: year, month, week_of_year, day_of_week, is_weekend, quarter,
    season (Spring/Summer/Autumn/Winter), is_december, is_january,
    days_to_christmas (for Nov-Dec), fiscal_year, fiscal_quarter,
    month_sin, month_cos (cyclic encoding), week_sin, week_cos
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"] = df[date_col].dt.quarter

    def get_season(month: int) -> str:
        if month in (3, 4, 5):
            return "Spring"
        elif month in (6, 7, 8):
            return "Summer"
        elif month in (9, 10, 11):
            return "Autumn"
        else:
            return "Winter"

    df["season"] = df["month"].apply(get_season)
    df["is_december"] = (df["month"] == 12).astype(int)
    df["is_january"] = (df["month"] == 1).astype(int)

    def days_to_christmas(row: pd.Series) -> int:
        if row["month"] in (11, 12):
            christmas = pd.Timestamp(year=row["year"], month=12, day=25)
            delta = (christmas - row[date_col]).days
            return max(0, int(delta))
        return -1

    df["days_to_christmas"] = df.apply(days_to_christmas, axis=1)

    # UK fiscal year starts April 1
    df["fiscal_year"] = df[date_col].apply(
        lambda d: d.year if d.month >= 4 else d.year - 1
    )
    df["fiscal_quarter"] = df["month"].apply(
        lambda m: ((m - 4) % 12) // 3 + 1
    )

    # Cyclic encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    return df


def compute_seasonality_index(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "invoice_date",
    period: str = "week",
) -> pd.DataFrame:
    """
    Compute seasonality index: ratio of period average to overall average.
    Index > 1 means above-average period; < 1 means below-average.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if period == "week":
        df["_period"] = df[date_col].dt.isocalendar().week.astype(int)
    elif period == "month":
        df["_period"] = df[date_col].dt.month
    elif period == "quarter":
        df["_period"] = df[date_col].dt.quarter
    else:
        raise ValueError(f"Unsupported period: {period}. Use 'week', 'month', or 'quarter'.")

    overall_mean = df[value_col].mean()
    if overall_mean == 0:
        logger.warning("Overall mean is zero; returning index of 1.0 everywhere.")
        period_means = df.groupby("_period")[value_col].mean().reset_index()
        period_means["seasonality_index"] = 1.0
    else:
        period_means = df.groupby("_period")[value_col].mean().reset_index()
        period_means["seasonality_index"] = period_means[value_col] / overall_mean

    period_means = period_means.rename(columns={"_period": period})
    period_means = period_means.drop(columns=[value_col])
    return period_means


def compute_holiday_lift(
    df: pd.DataFrame,
    value_col: str,
    date_col: str,
    bank_holidays_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute lift in value_col during bank holiday periods vs baseline.
    Returns DataFrame with: holiday_name, avg_baseline, avg_holiday, lift_ratio
    """
    if bank_holidays_df is None:
        bank_holidays_df = get_demo_bank_holidays()

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    bank_holidays_df = bank_holidays_df.copy()
    bank_holidays_df["date"] = pd.to_datetime(bank_holidays_df["date"])

    results = []
    for _, holiday in bank_holidays_df.iterrows():
        hdate = holiday["date"]
        hname = holiday["name"]

        holiday_mask = (df[date_col] >= hdate - pd.Timedelta(days=3)) & (
            df[date_col] <= hdate + pd.Timedelta(days=3)
        )
        baseline_mask = (df[date_col] >= hdate - pd.Timedelta(days=31)) & (
            df[date_col] < hdate - pd.Timedelta(days=3)
        )

        holiday_vals = df.loc[holiday_mask, value_col]
        baseline_vals = df.loc[baseline_mask, value_col]

        if len(holiday_vals) == 0 or len(baseline_vals) == 0:
            continue

        avg_holiday = holiday_vals.mean()
        avg_baseline = baseline_vals.mean()
        lift_ratio = avg_holiday / avg_baseline if avg_baseline != 0 else np.nan

        results.append(
            {
                "holiday_name": hname,
                "holiday_date": hdate,
                "avg_baseline": round(avg_baseline, 4),
                "avg_holiday": round(avg_holiday, 4),
                "lift_ratio": round(lift_ratio, 4) if not np.isnan(lift_ratio) else np.nan,
            }
        )

    return pd.DataFrame(results)


def get_demo_bank_holidays() -> pd.DataFrame:
    """Return hardcoded UK bank holidays 2020-2024 as DataFrame with date, name columns."""
    holidays = [
        # 2020
        ("2020-01-01", "New Year's Day"),
        ("2020-04-10", "Good Friday"),
        ("2020-04-13", "Easter Monday"),
        ("2020-05-08", "Early May Bank Holiday (VE Day)"),
        ("2020-05-25", "Spring Bank Holiday"),
        ("2020-08-31", "Summer Bank Holiday"),
        ("2020-12-25", "Christmas Day"),
        ("2020-12-28", "Boxing Day (substitute)"),
        # 2021
        ("2021-01-01", "New Year's Day"),
        ("2021-04-02", "Good Friday"),
        ("2021-04-05", "Easter Monday"),
        ("2021-05-03", "Early May Bank Holiday"),
        ("2021-05-31", "Spring Bank Holiday"),
        ("2021-08-30", "Summer Bank Holiday"),
        ("2021-12-27", "Christmas Day (substitute)"),
        ("2021-12-28", "Boxing Day (substitute)"),
        # 2022
        ("2022-01-03", "New Year's Day (substitute)"),
        ("2022-04-15", "Good Friday"),
        ("2022-04-18", "Easter Monday"),
        ("2022-05-02", "Early May Bank Holiday"),
        ("2022-06-02", "Spring Bank Holiday"),
        ("2022-06-03", "Platinum Jubilee"),
        ("2022-08-29", "Summer Bank Holiday"),
        ("2022-09-19", "State Funeral of Queen Elizabeth II"),
        ("2022-12-26", "Christmas Day (substitute)"),
        ("2022-12-27", "Boxing Day (substitute)"),
        # 2023
        ("2023-01-02", "New Year's Day (substitute)"),
        ("2023-04-07", "Good Friday"),
        ("2023-04-10", "Easter Monday"),
        ("2023-05-01", "Early May Bank Holiday"),
        ("2023-05-08", "Coronation of King Charles III"),
        ("2023-05-29", "Spring Bank Holiday"),
        ("2023-08-28", "Summer Bank Holiday"),
        ("2023-12-25", "Christmas Day"),
        ("2023-12-26", "Boxing Day"),
        # 2024
        ("2024-01-01", "New Year's Day"),
        ("2024-03-29", "Good Friday"),
        ("2024-04-01", "Easter Monday"),
        ("2024-05-06", "Early May Bank Holiday"),
        ("2024-05-27", "Spring Bank Holiday"),
        ("2024-08-26", "Summer Bank Holiday"),
        ("2024-12-25", "Christmas Day"),
        ("2024-12-26", "Boxing Day"),
    ]
    return pd.DataFrame(holidays, columns=["date", "name"])
