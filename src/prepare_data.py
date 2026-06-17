"""Prepare NESO demand data for modelling-ready daily analysis.

This module keeps the raw half-hourly data intact. It audits duplicate
settlement timestamps, resolves them with transparent aggregation rules, and
creates a daily dataset suitable for the first baseline modelling phase.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .eda import create_neso_datetime, standardise_column_names
    from .utils import PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR, TABLES_DIR, ensure_dir, load_json
except ImportError:  # Allows `python src/prepare_data.py` from the project root.
    from eda import create_neso_datetime, standardise_column_names
    from utils import PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR, TABLES_DIR, ensure_dir, load_json

SELECTED_RESOURCE_INFO_PATH = RAW_DATA_DIR / "selected_resource_info.json"
DUPLICATE_AUDIT_PATH = TABLES_DIR / "duplicate_settlement_datetime_audit.csv"
DEFAULT_DAILY_OUTPUT_PATH = PROCESSED_DATA_DIR / "daily_demand_2019_2025.csv"
TIMESTAMP_COLUMN = "settlement_datetime"
DATE_COLUMN = "settlement_date"
SETTLEMENT_PERIOD_COLUMN = "settlement_period"
TARGET_CANDIDATES = ["nd", "tsd", "england_wales_demand"]
EXTERNAL_MEAN_TERMS = [
    "embedded_wind",
    "embedded_solar",
    "pumped",
    "pump",
    "interconnector",
    "ifa",
    "britned",
    "moyle",
    "east_west",
    "nemo",
    "nsl",
    "eleclink",
    "viking",
]


def _project_path(path_value: str | Path) -> Path:
    """Resolve a repository-relative path against the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_selected_raw_data(selected_info_path: Path = SELECTED_RESOURCE_INFO_PATH) -> pd.DataFrame:
    """Load the combined raw dataset referenced by selected resource metadata."""
    selected_info = load_json(selected_info_path)
    combined_output_path = selected_info.get("combined_output_path")
    if not combined_output_path:
        raise KeyError(f"`combined_output_path` is missing from {selected_info_path}")

    raw_path = _project_path(combined_output_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Combined raw dataset not found at {raw_path}. Run `python src/ingest_neso.py` locally first."
        )

    df = pd.read_csv(raw_path)
    return standardise_column_names(df)


def ensure_settlement_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Create or validate `settlement_datetime` for half-hourly NESO records."""
    output = df.copy()
    if TIMESTAMP_COLUMN not in output.columns:
        return create_neso_datetime(output, DATE_COLUMN, SETTLEMENT_PERIOD_COLUMN, TIMESTAMP_COLUMN)

    output[TIMESTAMP_COLUMN] = pd.to_datetime(output[TIMESTAMP_COLUMN], errors="coerce")
    if output[TIMESTAMP_COLUMN].isna().any():
        if {DATE_COLUMN, SETTLEMENT_PERIOD_COLUMN}.issubset(output.columns):
            recreated = create_neso_datetime(output.drop(columns=[TIMESTAMP_COLUMN]), DATE_COLUMN, SETTLEMENT_PERIOD_COLUMN)
            output[TIMESTAMP_COLUMN] = output[TIMESTAMP_COLUMN].fillna(recreated[TIMESTAMP_COLUMN])
        if output[TIMESTAMP_COLUMN].isna().any():
            raise ValueError("`settlement_datetime` contains unparseable values after validation.")
    return output


def inspect_duplicate_timestamps(
    df: pd.DataFrame,
    datetime_column: str = TIMESTAMP_COLUMN,
    output_path: Path = DUPLICATE_AUDIT_PATH,
) -> pd.DataFrame:
    """Create and save an audit table for duplicated settlement timestamps."""
    duplicate_mask = df.duplicated(subset=[datetime_column], keep=False)
    duplicate_rows = df.loc[duplicate_mask].copy()
    if duplicate_rows.empty:
        audit_columns = [
            datetime_column,
            DATE_COLUMN,
            SETTLEMENT_PERIOD_COLUMN,
            "source_year",
            "nd",
            "tsd",
            "england_wales_demand",
            "duplicate_group_count",
            "duplicate_row_number",
        ]
        audit = pd.DataFrame(columns=[column for column in audit_columns if column in df.columns or column.startswith("duplicate")])
    else:
        duplicate_rows["duplicate_group_count"] = duplicate_rows.groupby(datetime_column)[datetime_column].transform("size")
        duplicate_rows["duplicate_row_number"] = duplicate_rows.groupby(datetime_column).cumcount() + 1
        preferred_columns = [
            datetime_column,
            DATE_COLUMN,
            SETTLEMENT_PERIOD_COLUMN,
            "source_year",
            "nd",
            "tsd",
            "england_wales_demand",
            "duplicate_group_count",
            "duplicate_row_number",
        ]
        audit = duplicate_rows[[column for column in preferred_columns if column in duplicate_rows.columns]]
        audit = audit.sort_values([datetime_column, "duplicate_row_number"])

    ensure_dir(output_path.parent)
    audit.to_csv(output_path, index=False)
    return audit


def _first_non_null(series: pd.Series) -> Any:
    """Return the first non-null value from a grouped series."""
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else np.nan


def resolve_duplicate_timestamps(df: pd.DataFrame, datetime_column: str = TIMESTAMP_COLUMN) -> pd.DataFrame:
    """Aggregate duplicate settlement timestamps without silently dropping data.

    Demand and other numeric measurement columns are averaged. Metadata columns
    that identify the settlement record use the first non-null value. A
    `duplicate_count` column records how many half-hourly records contributed to
    each resolved timestamp.
    """
    if datetime_column not in df.columns:
        raise KeyError(f"`{datetime_column}` is missing from the dataframe.")

    output = df.copy()
    output[datetime_column] = pd.to_datetime(output[datetime_column], errors="coerce")
    if output[datetime_column].isna().any():
        raise ValueError(f"`{datetime_column}` contains missing or invalid timestamps.")

    metadata_columns = {datetime_column, DATE_COLUMN, SETTLEMENT_PERIOD_COLUMN, "source_year"}
    aggregation: dict[str, str | Any] = {}
    for column in output.columns:
        if column == datetime_column:
            continue
        if column in metadata_columns:
            aggregation[column] = _first_non_null
        elif pd.api.types.is_numeric_dtype(output[column]):
            aggregation[column] = "mean"
        else:
            aggregation[column] = _first_non_null

    duplicate_counts = output.groupby(datetime_column).size().rename("duplicate_count").reset_index()
    resolved = output.groupby(datetime_column, as_index=False).agg(aggregation)
    resolved = resolved.merge(duplicate_counts, on=datetime_column, how="left")
    return resolved.sort_values(datetime_column).reset_index(drop=True)


def _last_sunday(year: int, month: int) -> date:
    """Return the date of the last Sunday in a month."""
    month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    offset_days = (month_end.weekday() + 1) % 7
    return (month_end - pd.Timedelta(days=offset_days)).date()


def expected_settlement_period_count(day: date) -> int:
    """Return expected GB settlement periods for a calendar day."""
    if day == _last_sunday(day.year, 3):
        return 46
    if day == _last_sunday(day.year, 10):
        return 50
    return 48


def _daily_mean_columns(df: pd.DataFrame) -> list[str]:
    """Identify external numeric columns to carry as daily means."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = set(TARGET_CANDIDATES + [SETTLEMENT_PERIOD_COLUMN, "source_year", "duplicate_count"])
    columns: list[str] = []
    for column in numeric_columns:
        column_lower = column.lower()
        if column in excluded:
            continue
        if any(term in column_lower for term in EXTERNAL_MEAN_TERMS):
            columns.append(column)
    return columns


def create_daily_demand_dataset(
    df: pd.DataFrame,
    target: str = "nd",
    datetime_column: str = TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    """Aggregate de-duplicated half-hourly NESO data to daily modelling rows."""
    if target not in df.columns:
        raise KeyError(f"Requested target `{target}` is not present in the dataframe.")
    if datetime_column not in df.columns:
        raise KeyError(f"`{datetime_column}` is missing from the dataframe.")

    working = df.copy()
    working[datetime_column] = pd.to_datetime(working[datetime_column], errors="coerce")
    working = working.dropna(subset=[datetime_column])
    working["date"] = working[datetime_column].dt.date

    named_aggregations: dict[str, tuple[str, str]] = {}
    for column in TARGET_CANDIDATES:
        if column in working.columns:
            named_aggregations[f"{column}_mean"] = (column, "mean")
            named_aggregations[f"{column}_peak"] = (column, "max")
    for column in _daily_mean_columns(working):
        named_aggregations[f"{column}_mean"] = (column, "mean")

    daily = working.groupby("date").agg(**named_aggregations).reset_index()
    coverage = (
        working.groupby("date")
        .agg(settlement_period_count=(datetime_column, "nunique"))
        .reset_index()
    )
    daily = daily.merge(coverage, on="date", how="left")

    date_index = pd.to_datetime(daily["date"])
    daily["expected_settlement_period_count"] = [expected_settlement_period_count(day.date()) for day in date_index]
    daily["coverage_ratio"] = daily["settlement_period_count"] / daily["expected_settlement_period_count"]
    daily["has_incomplete_day"] = daily["coverage_ratio"] < 1
    daily["year"] = date_index.dt.year
    daily["month"] = date_index.dt.month
    daily["day"] = date_index.dt.day
    daily["day_of_week"] = date_index.dt.dayofweek
    daily["week_of_year"] = date_index.dt.isocalendar().week.astype(int)
    daily["quarter"] = date_index.dt.quarter
    daily["is_weekend"] = daily["day_of_week"] >= 5

    leading_columns = [
        "date",
        "year",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "quarter",
        "is_weekend",
        "settlement_period_count",
        "expected_settlement_period_count",
        "coverage_ratio",
        "has_incomplete_day",
    ]
    remaining_columns = [column for column in daily.columns if column not in leading_columns]
    return daily[leading_columns + remaining_columns].sort_values("date").reset_index(drop=True)


def save_processed_outputs(
    daily_df: pd.DataFrame,
    output_path: Path = DEFAULT_DAILY_OUTPUT_PATH,
) -> Path:
    """Save the processed daily dataset."""
    ensure_dir(output_path.parent)
    daily_df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse data-preparation command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare NESO daily demand data from combined half-hourly raw data.")
    parser.add_argument("--target", default="nd", help="Default modelling target to validate. All major targets are retained.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DAILY_OUTPUT_PATH, help="Processed daily CSV output path.")
    return parser.parse_args()


def main() -> None:
    """Run the NESO data-preparation pipeline."""
    args = parse_args()
    output_path = _project_path(args.output)
    df = ensure_settlement_datetime(load_selected_raw_data())
    duplicate_audit = inspect_duplicate_timestamps(df)
    resolved = resolve_duplicate_timestamps(df)
    daily = create_daily_demand_dataset(resolved, target=args.target)
    saved_path = save_processed_outputs(daily, output_path)

    duplicate_row_count = len(duplicate_audit)
    duplicate_group_count = duplicate_audit[TIMESTAMP_COLUMN].nunique() if not duplicate_audit.empty else 0
    print(f"Duplicate timestamp audit rows: {duplicate_row_count} across {duplicate_group_count} timestamp group(s).")
    print(f"Saved duplicate audit to {DUPLICATE_AUDIT_PATH}")
    print(f"Saved processed daily dataset to {saved_path}")
    print(f"Daily rows: {len(daily)}")
    print(f"Incomplete days flagged: {int(daily['has_incomplete_day'].sum())}")


if __name__ == "__main__":
    main()
