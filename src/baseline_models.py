"""Baseline demand forecasting models for the prepared NESO daily dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from .utils import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR, ensure_dir
except ImportError:  # Allows `python src/baseline_models.py` from the project root.
    from utils import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR, ensure_dir

DEFAULT_DAILY_DATA_PATH = PROCESSED_DATA_DIR / "daily_demand_2019_2025.csv"
DEFAULT_TARGET = "nd_mean"
DEFAULT_TEST_START = "2025-01-01"
DEFAULT_TEST_END = "2025-12-31"
MODEL_COMPARISON_PATH = TABLES_DIR / "baseline_model_comparison.csv"
BASELINE_FORECASTS_PATH = TABLES_DIR / "baseline_forecasts.csv"
MODELLING_FIGURES_DIR = FIGURES_DIR / "modelling"


def _project_path(path_value: str | Path) -> Path:
    """Resolve repository-relative paths against the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_daily_dataset(path: str | Path = DEFAULT_DAILY_DATA_PATH) -> pd.DataFrame:
    """Load the processed daily demand dataset."""
    data_path = _project_path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed daily dataset not found at {data_path}. "
            "Run `python src/prepare_data.py` locally before baseline modelling."
        )
    df = pd.read_csv(data_path)
    if "date" not in df.columns:
        raise KeyError("Processed daily dataset must include a `date` column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Processed daily dataset contains unparseable dates.")
    return df.sort_values("date").reset_index(drop=True)


def create_time_series(df: pd.DataFrame, target: str = DEFAULT_TARGET) -> pd.Series:
    """Create a daily target series indexed by date."""
    if target not in df.columns:
        raise KeyError(f"Target `{target}` is not present in the processed dataset.")
    series = df[["date", target]].dropna().sort_values("date").set_index("date")[target]
    series = series.asfreq("D")
    return series


def train_test_split_time_series(
    series: pd.Series,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> tuple[pd.Series, pd.Series]:
    """Split a time series chronologically with no shuffling."""
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    train = series.loc[series.index < start].dropna()
    test = series.loc[(series.index >= start) & (series.index <= end)].dropna()
    if train.empty:
        raise ValueError("Training series is empty. Choose an earlier test start date.")
    if test.empty:
        raise ValueError("Test series is empty. Check the processed dataset and test date range.")
    return train, test


def naive_forecast(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """Forecast each test date as the last observed training value."""
    last_value = float(train.dropna().iloc[-1])
    return pd.Series(last_value, index=test_index, name="naive_forecast")


def seasonal_naive_forecast(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """Forecast each date using the same calendar day from the previous year where available."""
    historical = train.dropna()
    fallback_value = float(historical.iloc[-1])
    forecast_values: list[float] = []
    for timestamp in test_index:
        previous_year_timestamp = timestamp - pd.DateOffset(years=1)
        if previous_year_timestamp in historical.index:
            forecast_values.append(float(historical.loc[previous_year_timestamp]))
        else:
            lagged_position = historical.index.get_indexer([timestamp - pd.Timedelta(days=365)])[0]
            forecast_values.append(float(historical.iloc[lagged_position]) if lagged_position >= 0 else fallback_value)
    return pd.Series(forecast_values, index=test_index, name="seasonal_naive_forecast")


def holt_winters_forecast(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """Forecast with Holt-Winters Exponential Smoothing, falling back gracefully if needed."""
    clean_train = train.dropna().asfreq("D")
    horizon = len(test_index)
    model_attempts: list[dict[str, Any]] = []
    if len(clean_train) >= 730:
        model_attempts.append({"trend": "add", "seasonal": "add", "seasonal_periods": 365})
    model_attempts.append({"trend": "add", "seasonal": None, "seasonal_periods": None})
    model_attempts.append({"trend": None, "seasonal": None, "seasonal_periods": None})

    for params in model_attempts:
        try:
            model = ExponentialSmoothing(
                clean_train,
                trend=params["trend"],
                seasonal=params["seasonal"],
                seasonal_periods=params["seasonal_periods"],
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)
            forecast = fitted.forecast(horizon)
            return pd.Series(forecast.to_numpy(), index=test_index, name="holt_winters_forecast")
        except Exception as exc:  # statsmodels can fail on missing dates or unstable seasonal fits.
            print(f"Holt-Winters attempt failed with {params}: {exc}")

    return naive_forecast(train, test_index).rename("holt_winters_forecast")


def evaluate_forecast(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    """Calculate MAE, RMSE and MAPE for one forecast series."""
    aligned_actual, aligned_forecast = actual.align(forecast, join="inner")
    errors = aligned_actual - aligned_forecast
    non_zero_actual = aligned_actual.replace(0, np.nan)
    return {
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": float((errors.abs() / non_zero_actual).mean() * 100),
    }


def _plot_actual_vs_forecasts(forecasts: pd.DataFrame, output_path: Path) -> None:
    """Save an actual-vs-forecast line plot."""
    ensure_dir(output_path.parent)
    plt.figure(figsize=(14, 6))
    plt.plot(forecasts["date"], forecasts["actual"], label="actual", linewidth=2)
    for column in ["naive_forecast", "seasonal_naive_forecast", "holt_winters_forecast"]:
        plt.plot(forecasts["date"], forecasts[column], label=column, linewidth=1)
    plt.title("Actual demand versus baseline forecasts")
    plt.xlabel("date")
    plt.ylabel("demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_forecast_errors(forecasts: pd.DataFrame, output_path: Path) -> None:
    """Save a boxplot of forecast errors by model."""
    ensure_dir(output_path.parent)
    error_columns = {
        "naive": forecasts["actual"] - forecasts["naive_forecast"],
        "seasonal_naive": forecasts["actual"] - forecasts["seasonal_naive_forecast"],
        "holt_winters": forecasts["actual"] - forecasts["holt_winters_forecast"],
    }
    plt.figure(figsize=(10, 6))
    plt.boxplot(error_columns.values(), labels=error_columns.keys(), showfliers=False)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Forecast errors by baseline model")
    plt.ylabel("actual - forecast")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_baseline_forecasting(
    input_path: str | Path = DEFAULT_DAILY_DATA_PATH,
    target: str = DEFAULT_TARGET,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all baseline models, save forecasts, metrics and diagnostic figures."""
    df = load_daily_dataset(input_path)
    series = create_time_series(df, target=target)
    train, test = train_test_split_time_series(series, test_start=test_start, test_end=test_end)

    forecasts = pd.DataFrame(
        {
            "date": test.index,
            "actual": test.to_numpy(),
            "naive_forecast": naive_forecast(train, test.index).to_numpy(),
            "seasonal_naive_forecast": seasonal_naive_forecast(train, test.index).to_numpy(),
            "holt_winters_forecast": holt_winters_forecast(train, test.index).to_numpy(),
        }
    )

    comparison_rows = []
    for model_name, column in [
        ("naive", "naive_forecast"),
        ("seasonal_naive", "seasonal_naive_forecast"),
        ("holt_winters", "holt_winters_forecast"),
    ]:
        metrics = evaluate_forecast(pd.Series(forecasts["actual"].to_numpy(), index=test.index), pd.Series(forecasts[column].to_numpy(), index=test.index))
        comparison_rows.append({"model": model_name, **metrics})
    comparison = pd.DataFrame(comparison_rows).sort_values("mae").reset_index(drop=True)

    ensure_dir(TABLES_DIR)
    forecasts.to_csv(BASELINE_FORECASTS_PATH, index=False)
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    _plot_actual_vs_forecasts(forecasts, MODELLING_FIGURES_DIR / "actual_vs_baseline_forecasts.png")
    _plot_forecast_errors(forecasts, MODELLING_FIGURES_DIR / "forecast_errors_by_model.png")
    return comparison, forecasts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline forecasting."""
    parser = argparse.ArgumentParser(description="Run baseline forecasts for the NESO daily demand dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DAILY_DATA_PATH, help="Processed daily dataset path.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column to forecast.")
    parser.add_argument("--test-start", default=DEFAULT_TEST_START, help="First date in the test set.")
    parser.add_argument("--test-end", default=DEFAULT_TEST_END, help="Last date in the test set.")
    return parser.parse_args()


def main() -> None:
    """Run the baseline forecasting CLI."""
    args = parse_args()
    comparison, forecasts = run_baseline_forecasting(
        input_path=args.input,
        target=args.target,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    print("Baseline model comparison:")
    print(comparison)
    print(f"Saved model comparison to {MODEL_COMPARISON_PATH}")
    print(f"Saved forecasts to {BASELINE_FORECASTS_PATH}")
    print(f"Saved modelling figures to {MODELLING_FIGURES_DIR}")
    print(f"Forecast rows: {len(forecasts)}")


if __name__ == "__main__":
    main()
