"""Statistical demand forecasting models for the prepared NESO daily dataset."""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from .utils import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR, ensure_dir
except ImportError:  # Allows `python src/statistical_models.py` from the project root.
    from utils import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR, ensure_dir

DEFAULT_DAILY_DATA_PATH = PROCESSED_DATA_DIR / "daily_demand_2019_2025.csv"
DEFAULT_TARGET = "nd_mean"
DEFAULT_TEST_START = "2025-01-01"
DEFAULT_TEST_END = "2025-12-31"
BASELINE_COMPARISON_PATH = TABLES_DIR / "baseline_model_comparison.csv"
BASELINE_FORECASTS_PATH = TABLES_DIR / "baseline_forecasts.csv"
STATISTICAL_COMPARISON_PATH = TABLES_DIR / "statistical_model_comparison.csv"
STATISTICAL_FORECASTS_PATH = TABLES_DIR / "statistical_forecasts.csv"
MODELLING_FIGURES_DIR = FIGURES_DIR / "modelling"
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 7)
EXOGENOUS_CANDIDATES = [
    "embedded_wind_generation_mean",
    "embedded_solar_generation_mean",
    "pump_storage_pumping_mean",
    "ifa_flow_mean",
    "ifa2_flow_mean",
    "britned_flow_mean",
    "moyle_flow_mean",
    "east_west_flow_mean",
    "nemo_flow_mean",
    "nsl_flow_mean",
    "eleclink_flow_mean",
    "viking_flow_mean",
    "is_weekend",
    "month",
    "day_of_week",
]


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
            "Run `python src/prepare_data.py` locally before statistical modelling."
        )
    df = pd.read_csv(data_path)
    if "date" not in df.columns:
        raise KeyError("Processed daily dataset must include a `date` column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Processed daily dataset contains unparseable dates.")
    return df.sort_values("date").reset_index(drop=True)


def load_baseline_results(
    comparison_path: str | Path = BASELINE_COMPARISON_PATH,
    forecasts_path: str | Path = BASELINE_FORECASTS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load baseline comparison and forecast outputs where available."""
    comparison_file = _project_path(comparison_path)
    forecasts_file = _project_path(forecasts_path)
    comparison = pd.read_csv(comparison_file) if comparison_file.exists() else pd.DataFrame()
    forecasts = pd.read_csv(forecasts_file) if forecasts_file.exists() else pd.DataFrame()
    if "date" in forecasts.columns:
        forecasts["date"] = pd.to_datetime(forecasts["date"], errors="coerce")
    return comparison, forecasts


def create_target_series(df: pd.DataFrame, target: str = DEFAULT_TARGET) -> pd.Series:
    """Create a daily target series indexed by date."""
    if target not in df.columns:
        raise KeyError(f"Target `{target}` is not present in the processed dataset.")
    return df[["date", target]].dropna().sort_values("date").set_index("date")[target].asfreq("D")


def select_exogenous_features(df: pd.DataFrame, candidates: list[str] | None = None) -> pd.DataFrame:
    """Select available SARIMAX exogenous features and fill missing values safely."""
    candidates = candidates or EXOGENOUS_CANDIDATES
    available_columns = [column for column in candidates if column in df.columns]
    if not available_columns:
        return pd.DataFrame(index=pd.to_datetime(df["date"]))

    exog = df[["date", *available_columns]].copy()
    exog["date"] = pd.to_datetime(exog["date"], errors="coerce")
    exog = exog.dropna(subset=["date"]).sort_values("date").set_index("date")
    for column in available_columns:
        exog[column] = pd.to_numeric(exog[column], errors="coerce")
    return exog.ffill().bfill().fillna(0)


def train_test_split(
    series: pd.Series,
    exog: pd.DataFrame | None = None,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> tuple[pd.Series, pd.Series, pd.DataFrame | None, pd.DataFrame | None]:
    """Split target and optional exogenous features chronologically."""
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    train_y = series.loc[series.index < start].dropna()
    test_y = series.loc[(series.index >= start) & (series.index <= end)].dropna()
    if train_y.empty:
        raise ValueError("Training series is empty. Choose an earlier test start date.")
    if test_y.empty:
        raise ValueError("Test series is empty. Check the processed dataset and test date range.")

    if exog is None or exog.empty:
        return train_y, test_y, None, None
    aligned_exog = exog.reindex(series.index).ffill().bfill().fillna(0)
    train_exog = aligned_exog.loc[train_y.index]
    test_exog = aligned_exog.loc[test_y.index]
    train_mean = train_exog.mean()
    train_std = train_exog.std().replace(0, 1)
    train_exog = (train_exog - train_mean) / train_std
    test_exog = (test_exog - train_mean) / train_std
    return train_y, test_y, train_exog, test_exog


def fit_sarima(
    train: pd.Series,
    order: tuple[int, int, int] = SARIMA_ORDER,
    seasonal_order: tuple[int, int, int, int] = SARIMA_SEASONAL_ORDER,
) -> Any:
    """Fit a practical daily SARIMA model with weekly seasonality."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=200)


def fit_sarimax(
    train: pd.Series,
    train_exog: pd.DataFrame,
    order: tuple[int, int, int] = SARIMA_ORDER,
    seasonal_order: tuple[int, int, int, int] = SARIMA_SEASONAL_ORDER,
) -> Any:
    """Fit SARIMAX with selected exogenous features."""
    if train_exog.empty:
        raise ValueError("SARIMAX requires at least one exogenous feature.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = SARIMAX(
            train,
            exog=train_exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=200)


def forecast_sarima(fitted_model: Any, test_index: pd.DatetimeIndex) -> pd.Series:
    """Forecast SARIMA values for the test index."""
    forecast = fitted_model.forecast(steps=len(test_index))
    return pd.Series(forecast.to_numpy(), index=test_index, name="sarima_forecast")


def forecast_sarimax(fitted_model: Any, test_exog: pd.DataFrame, test_index: pd.DatetimeIndex) -> pd.Series:
    """Forecast SARIMAX values for the test index."""
    forecast = fitted_model.forecast(steps=len(test_index), exog=test_exog)
    return pd.Series(forecast.to_numpy(), index=test_index, name="sarimax_forecast")


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


def _plot_actual_vs_statistical_forecasts(forecasts: pd.DataFrame, output_path: Path) -> None:
    """Save actual, seasonal naive and statistical forecast lines."""
    ensure_dir(output_path.parent)
    plt.figure(figsize=(14, 6))
    plt.plot(forecasts["date"], forecasts["actual"], label="actual", linewidth=2)
    for column in ["seasonal_naive_forecast", "sarima_forecast", "sarimax_forecast"]:
        if column in forecasts.columns:
            plt.plot(forecasts["date"], forecasts[column], label=column, linewidth=1)
    plt.title("Actual demand versus statistical forecasts")
    plt.xlabel("date")
    plt.ylabel("demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_statistical_errors(forecasts: pd.DataFrame, output_path: Path) -> None:
    """Save a boxplot of statistical forecast errors."""
    ensure_dir(output_path.parent)
    error_columns = {
        "sarima": forecasts["actual"] - forecasts["sarima_forecast"],
        "sarimax": forecasts["actual"] - forecasts["sarimax_forecast"],
    }
    if "seasonal_naive_forecast" in forecasts.columns:
        error_columns = {"seasonal_naive": forecasts["actual"] - forecasts["seasonal_naive_forecast"], **error_columns}
    labels = list(error_columns.keys())
    plt.figure(figsize=(10, 6))
    plt.boxplot(list(error_columns.values()), showfliers=False)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Forecast errors by statistical model")
    plt.ylabel("actual - forecast")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_mape_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    """Save a bar chart comparing model MAPE."""
    ensure_dir(output_path.parent)
    plot_df = comparison.dropna(subset=["mape"]).sort_values("mape")
    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["model"], plot_df["mape"])
    plt.title("Model MAPE comparison")
    plt.xlabel("model")
    plt.ylabel("MAPE (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _combine_comparison(baseline_comparison: pd.DataFrame, statistical_rows: list[dict[str, float | str]]) -> pd.DataFrame:
    """Combine baseline and statistical metrics in a stable model order."""
    expected_baselines = ["naive", "seasonal_naive", "holt_winters"]
    if baseline_comparison.empty:
        baseline = pd.DataFrame(columns=["model", "mae", "rmse", "mape"])
    else:
        baseline = baseline_comparison[baseline_comparison["model"].isin(expected_baselines)].copy()
    comparison = pd.concat([baseline, pd.DataFrame(statistical_rows)], ignore_index=True)
    order = {model: idx for idx, model in enumerate([*expected_baselines, "sarima", "sarimax"])}
    comparison["model_order"] = comparison["model"].map(order)
    return comparison.sort_values("model_order").drop(columns=["model_order"]).reset_index(drop=True)


def run_statistical_forecasting(
    input_path: str | Path = DEFAULT_DAILY_DATA_PATH,
    target: str = DEFAULT_TARGET,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run SARIMA/SARIMAX forecasting and save comparisons, forecasts and plots."""
    df = load_daily_dataset(input_path)
    baseline_comparison, baseline_forecasts = load_baseline_results()
    series = create_target_series(df, target=target)
    exog = select_exogenous_features(df)
    train_y, test_y, train_exog, test_exog = train_test_split(series, exog, test_start, test_end)

    sarima_result = fit_sarima(train_y)
    sarima_forecast = forecast_sarima(sarima_result, test_y.index)

    if train_exog is not None and test_exog is not None and not train_exog.empty:
        try:
            sarimax_result = fit_sarimax(train_y, train_exog)
            sarimax_forecast = forecast_sarimax(sarimax_result, test_exog, test_y.index)
        except Exception as exc:
            print(f"SARIMAX failed and will fall back to SARIMA forecast: {exc}")
            sarimax_forecast = sarima_forecast.rename("sarimax_forecast")
    else:
        print("No configured exogenous features were available; SARIMAX will fall back to SARIMA forecast.")
        sarimax_forecast = sarima_forecast.rename("sarimax_forecast")

    forecasts = pd.DataFrame({"date": test_y.index, "actual": test_y.to_numpy()})
    if not baseline_forecasts.empty and "seasonal_naive_forecast" in baseline_forecasts.columns:
        seasonal_baseline = baseline_forecasts[["date", "seasonal_naive_forecast"]].copy()
        forecasts = forecasts.merge(seasonal_baseline, on="date", how="left")
    forecasts["sarima_forecast"] = sarima_forecast.to_numpy()
    forecasts["sarimax_forecast"] = sarimax_forecast.to_numpy()

    statistical_rows = [
        {"model": "sarima", **evaluate_forecast(test_y, sarima_forecast)},
        {"model": "sarimax", **evaluate_forecast(test_y, sarimax_forecast)},
    ]
    comparison = _combine_comparison(baseline_comparison, statistical_rows)

    ensure_dir(TABLES_DIR)
    forecasts.to_csv(STATISTICAL_FORECASTS_PATH, index=False)
    comparison.to_csv(STATISTICAL_COMPARISON_PATH, index=False)
    _plot_actual_vs_statistical_forecasts(forecasts, MODELLING_FIGURES_DIR / "actual_vs_statistical_forecasts.png")
    _plot_statistical_errors(forecasts, MODELLING_FIGURES_DIR / "statistical_forecast_errors.png")
    _plot_mape_comparison(comparison, MODELLING_FIGURES_DIR / "model_mape_comparison.png")

    seasonal_row = comparison.loc[comparison["model"] == "seasonal_naive"]
    if not seasonal_row.empty:
        seasonal_mae = float(seasonal_row["mae"].iloc[0])
        for model in ["sarima", "sarimax"]:
            model_mae = float(comparison.loc[comparison["model"] == model, "mae"].iloc[0])
            verdict = "beats" if model_mae < seasonal_mae else "does not beat"
            print(f"{model} {verdict} seasonal_naive on MAE ({model_mae:.2f} vs {seasonal_mae:.2f}).")

    print(f"SARIMAX exogenous features used: {list(exog.columns)}")
    return comparison, forecasts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for statistical forecasting."""
    parser = argparse.ArgumentParser(description="Run SARIMA and SARIMAX forecasts for NESO daily demand.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DAILY_DATA_PATH, help="Processed daily dataset path.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column to forecast.")
    parser.add_argument("--test-start", default=DEFAULT_TEST_START, help="First date in the test set.")
    parser.add_argument("--test-end", default=DEFAULT_TEST_END, help="Last date in the test set.")
    return parser.parse_args()


def main() -> None:
    """Run the statistical forecasting CLI."""
    args = parse_args()
    comparison, forecasts = run_statistical_forecasting(
        input_path=args.input,
        target=args.target,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    print("Statistical model comparison:")
    print(comparison)
    print(f"Saved statistical comparison to {STATISTICAL_COMPARISON_PATH}")
    print(f"Saved statistical forecasts to {STATISTICAL_FORECASTS_PATH}")
    print(f"Forecast rows: {len(forecasts)}")


if __name__ == "__main__":
    main()
