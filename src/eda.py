"""Reusable exploratory data analysis helpers for NESO demand time series."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

try:
    from .utils import ensure_dir
except ImportError:
    from utils import ensure_dir


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load a raw CSV or Excel file without modifying the source file."""
    file_path = Path(path)
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with conservative snake_case column names."""
    output = df.copy()
    output.columns = (
        pd.Index(output.columns.astype(str))
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.strip("_")
    )
    return output


def summarise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise column types, non-null counts, uniqueness, and sample values."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "non_null_count": [df[col].notna().sum() for col in df.columns],
            "missing_count": [df[col].isna().sum() for col in df.columns],
            "missing_pct": [df[col].isna().mean() * 100 for col in df.columns],
            "unique_count": [df[col].nunique(dropna=True) for col in df.columns],
            "example_value": [df[col].dropna().iloc[0] if df[col].notna().any() else np.nan for col in df.columns],
        }
    )


def detect_datetime_columns(df: pd.DataFrame, sample_size: int = 1000) -> list[str]:
    """Identify columns that are already datetime typed or parse as dates in a sample."""
    candidates: list[str] = []
    for column in df.columns:
        series = df[column].dropna().head(sample_size)
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            candidates.append(column)
        elif not series.empty:
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
            if parsed.notna().mean() >= 0.8:
                candidates.append(column)
    return candidates


def parse_datetime_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy with one column parsed to pandas datetime."""
    output = df.copy()
    output[column] = pd.to_datetime(output[column], errors="coerce")
    return output


def infer_time_frequency(df: pd.DataFrame, datetime_column: str) -> dict[str, Any]:
    """Infer time-step information from sorted timestamp differences."""
    timestamps = pd.to_datetime(df[datetime_column], errors="coerce").dropna().sort_values().drop_duplicates()
    diffs = timestamps.diff().dropna()
    return {
        "pandas_inferred_frequency": pd.infer_freq(timestamps) if len(timestamps) >= 3 else None,
        "most_common_interval": diffs.mode().iloc[0] if not diffs.empty else None,
        "minimum_interval": diffs.min() if not diffs.empty else None,
        "maximum_interval": diffs.max() if not diffs.empty else None,
    }


def find_duplicate_timestamps(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """Return all rows whose timestamp appears more than once."""
    return df[df.duplicated(subset=[datetime_column], keep=False)].sort_values(datetime_column)


def summarise_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing values by column."""
    missing = df.isna().sum().rename("missing_count").reset_index().rename(columns={"index": "column"})
    missing["missing_pct"] = missing["missing_count"] / len(df) * 100 if len(df) else 0
    return missing.sort_values(["missing_count", "column"], ascending=[False, True])


def identify_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def identify_candidate_demand_columns(df: pd.DataFrame) -> list[str]:
    """Find numeric columns with names suggesting electricity demand concepts."""
    terms = ["demand", "load", "nd", "tsd", "transmission", "national", "total"]
    numeric = identify_numeric_columns(df)
    return [col for col in numeric if any(term in col.lower() for term in terms)]


def _save_plot(output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_series(df: pd.DataFrame, datetime_column: str, target_column: str, output_path: str | Path) -> None:
    """Plot a target variable against time."""
    plot_df = df[[datetime_column, target_column]].dropna().sort_values(datetime_column)
    plt.figure(figsize=(14, 5))
    plt.plot(plot_df[datetime_column], plot_df[target_column], linewidth=0.8)
    plt.title(f"{target_column} over time")
    plt.xlabel(datetime_column)
    plt.ylabel(target_column)
    _save_plot(output_path)


def plot_missing_values(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot a missingness matrix for all columns."""
    plt.figure(figsize=(14, 6))
    sns.heatmap(df.isna(), cbar=False)
    plt.title("Missing values by row and column")
    _save_plot(output_path)


def plot_distribution(df: pd.DataFrame, column: str, output_path: str | Path) -> None:
    """Plot histogram and kernel density for a numeric column."""
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column].dropna(), kde=True, bins=50)
    plt.title(f"Distribution of {column}")
    _save_plot(output_path)


def _pattern_plot(df: pd.DataFrame, datetime_column: str, target_column: str, key: str, output_path: str | Path) -> None:
    plot_df = df[[datetime_column, target_column]].dropna().copy()
    plot_df[datetime_column] = pd.to_datetime(plot_df[datetime_column])
    accessors = {
        "hour": plot_df[datetime_column].dt.hour,
        "day_of_week": plot_df[datetime_column].dt.day_name(),
        "month": plot_df[datetime_column].dt.month_name(),
        "year": plot_df[datetime_column].dt.year,
    }
    plot_df[key] = accessors[key]
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=plot_df, x=key, y=target_column, estimator="mean", errorbar=None)
    plt.title(f"Average {target_column} by {key}")
    plt.xticks(rotation=45)
    _save_plot(output_path)


def plot_daily_pattern_if_applicable(df: pd.DataFrame, datetime_column: str, target_column: str, output_path: str | Path) -> None:
    """Plot average demand by hour when sub-daily timestamps are available."""
    _pattern_plot(df, datetime_column, target_column, "hour", output_path)


def plot_day_of_week_pattern(df: pd.DataFrame, datetime_column: str, target_column: str, output_path: str | Path) -> None:
    """Plot average demand by day of week."""
    _pattern_plot(df, datetime_column, target_column, "day_of_week", output_path)


def plot_monthly_pattern(df: pd.DataFrame, datetime_column: str, target_column: str, output_path: str | Path) -> None:
    """Plot average demand by month."""
    _pattern_plot(df, datetime_column, target_column, "month", output_path)


def plot_yearly_pattern(df: pd.DataFrame, datetime_column: str, target_column: str, output_path: str | Path) -> None:
    """Plot average demand by year."""
    _pattern_plot(df, datetime_column, target_column, "year", output_path)


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return rows outside the 1.5×IQR range for a numeric column."""
    series = df[column].dropna()
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df[(df[column] < lower) | (df[column] > upper)].copy()


def compute_correlation_table(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Compute numeric Pearson correlations against a target variable."""
    numeric = df.select_dtypes(include=[np.number])
    if target_column not in numeric.columns:
        return pd.DataFrame(columns=["column", "correlation_with_target"])
    return numeric.corr()[target_column].drop(target_column, errors="ignore").sort_values(key=np.abs, ascending=False).rename("correlation_with_target").reset_index().rename(columns={"index": "column"})


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot a correlation heatmap for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0)
    plt.title("Numeric correlation heatmap")
    _save_plot(output_path)


def run_stationarity_checks(df: pd.DataFrame, datetime_column: str, target_column: str) -> dict[str, Any]:
    """Run rolling statistics and Augmented Dickey-Fuller checks for a target."""
    series = df[[datetime_column, target_column]].dropna().sort_values(datetime_column)[target_column]
    window = min(48, max(2, len(series) // 20)) if len(series) else 2
    result: dict[str, Any] = {
        "rolling_window": window,
        "rolling_mean_latest": series.rolling(window).mean().dropna().iloc[-1] if len(series) >= window else np.nan,
        "rolling_std_latest": series.rolling(window).std().dropna().iloc[-1] if len(series) >= window else np.nan,
    }
    if len(series) >= 20:
        adf_stat, p_value, used_lag, n_obs, critical_values, icbest = adfuller(series, autolag="AIC")
        result.update({"adf_statistic": adf_stat, "adf_p_value": p_value, "adf_used_lag": used_lag, "adf_n_obs": n_obs, "adf_critical_values": critical_values, "adf_icbest": icbest})
    return result


def decompose_time_series(df: pd.DataFrame, datetime_column: str, target_column: str, output_path: str | Path, period: int | None = None) -> dict[str, Any]:
    """Perform additive seasonal decomposition and save the plot when feasible."""
    series = df[[datetime_column, target_column]].dropna().sort_values(datetime_column).set_index(datetime_column)[target_column]
    if period is None:
        period = 48 if len(series) >= 96 else 7 if len(series) >= 14 else None
    if period is None or len(series) < period * 2:
        return {"decomposed": False, "reason": "Insufficient observations for seasonal decomposition", "period": period}
    result = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    fig = result.plot()
    fig.set_size_inches(12, 8)
    _save_plot(output_path)
    return {"decomposed": True, "period": period}
