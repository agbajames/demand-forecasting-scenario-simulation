"""Diagnostics for baseline and statistical demand forecasting outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .baseline_models import evaluate_forecast, seasonal_naive_forecast
    from .utils import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR, ensure_dir
except ImportError:  # Allows `python src/model_diagnostics.py` from the project root.
    from baseline_models import evaluate_forecast, seasonal_naive_forecast
    from utils import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR, ensure_dir

DEFAULT_DAILY_DATA_PATH = PROCESSED_DATA_DIR / "daily_demand_2019_2025.csv"
BASELINE_FORECASTS_PATH = TABLES_DIR / "baseline_forecasts.csv"
STATISTICAL_FORECASTS_PATH = TABLES_DIR / "statistical_forecasts.csv"
STATISTICAL_COMPARISON_PATH = TABLES_DIR / "statistical_model_comparison.csv"
MODELLING_FIGURES_DIR = FIGURES_DIR / "modelling"
DEFAULT_TARGET = "nd_mean"
DEFAULT_TEST_START = "2025-01-01"
DEFAULT_TEST_END = "2025-12-31"
FORECAST_COLUMNS = [
    "seasonal_naive_forecast",
    "sarima_forecast",
    "sarimax_forecast",
    "holt_winters_forecast",
    "naive_forecast",
]
EXOGENOUS_CANDIDATES = [
    "embedded_wind_generation_mean",
    "embedded_solar_generation_mean",
    "embedded_wind_capacity_mean",
    "embedded_solar_capacity_mean",
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


def _read_csv_required(path: str | Path, description: str) -> pd.DataFrame:
    """Read a required generated CSV with a clear local-execution error."""
    csv_path = _project_path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {csv_path}. Run the ingestion, preparation, baseline and statistical "
            "forecasting commands locally before diagnostics."
        )
    return pd.read_csv(csv_path)


def load_daily_dataset(path: str | Path = DEFAULT_DAILY_DATA_PATH) -> pd.DataFrame:
    """Load the processed daily demand dataset."""
    df = _read_csv_required(path, "processed daily dataset")
    if "date" not in df.columns:
        raise KeyError("Processed daily dataset must include a `date` column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Processed daily dataset contains unparseable dates.")
    return df.sort_values("date").reset_index(drop=True)


def load_baseline_forecasts(path: str | Path = BASELINE_FORECASTS_PATH) -> pd.DataFrame:
    """Load baseline forecast outputs."""
    df = _read_csv_required(path, "baseline forecast output")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def load_statistical_forecasts(path: str | Path = STATISTICAL_FORECASTS_PATH) -> pd.DataFrame:
    """Load statistical forecast outputs."""
    df = _read_csv_required(path, "statistical forecast output")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def load_statistical_comparison(path: str | Path = STATISTICAL_COMPARISON_PATH) -> pd.DataFrame:
    """Load the combined statistical model comparison table."""
    return _read_csv_required(path, "statistical model comparison output")


def _merge_forecasts(baseline_forecasts: pd.DataFrame, statistical_forecasts: pd.DataFrame) -> pd.DataFrame:
    """Merge baseline and statistical forecast outputs on date and actual."""
    baseline_columns = ["date", *[column for column in ["naive_forecast", "holt_winters_forecast"] if column in baseline_forecasts.columns]]
    merged = statistical_forecasts.copy()
    if len(baseline_columns) > 1:
        merged = merged.merge(baseline_forecasts[baseline_columns], on="date", how="left")
    if "actual" not in merged.columns and "actual" in baseline_forecasts.columns:
        merged = merged.merge(baseline_forecasts[["date", "actual"]], on="date", how="left")
    return merged


def calculate_error_columns(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Add error, absolute error and absolute percentage error columns for each model."""
    if "actual" not in forecasts.columns:
        raise KeyError("Forecast outputs must include an `actual` column.")
    output = forecasts.copy()
    available_forecasts = [column for column in FORECAST_COLUMNS if column in output.columns]
    for forecast_column in available_forecasts:
        model_name = forecast_column.removesuffix("_forecast")
        output[f"{model_name}_error"] = output["actual"] - output[forecast_column]
        output[f"{model_name}_absolute_error"] = output[f"{model_name}_error"].abs()
        output[f"{model_name}_absolute_percentage_error"] = (
            output[f"{model_name}_absolute_error"] / output["actual"].replace(0, np.nan) * 100
        )
    return output


def _add_period_columns(errors: pd.DataFrame) -> pd.DataFrame:
    """Add month, quarter, weekday, weekend and demand-regime fields."""
    output = errors.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["month"] = output["date"].dt.month
    output["quarter"] = output["date"].dt.quarter
    output["day_of_week"] = output["date"].dt.dayofweek
    output["is_weekend"] = output["day_of_week"] >= 5
    q25, q75 = output["actual"].quantile([0.25, 0.75])
    output["demand_regime"] = np.select(
        [output["actual"] <= q25, output["actual"] >= q75],
        ["low", "high"],
        default="normal",
    )
    return output


def _summarise_group(errors: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Summarise model errors for one grouping column."""
    rows: list[dict[str, float | str | int | bool]] = []
    for forecast_column in [column for column in FORECAST_COLUMNS if column in errors.columns]:
        model = forecast_column.removesuffix("_forecast")
        grouped = errors.groupby(group_column, dropna=False)
        for group_value, group_df in grouped:
            rows.append(
                {
                    group_column: group_value,
                    "model": model,
                    "rows": len(group_df),
                    "mean_error": group_df[f"{model}_error"].mean(),
                    "mae": group_df[f"{model}_absolute_error"].mean(),
                    "mape": group_df[f"{model}_absolute_percentage_error"].mean(),
                }
            )
    return pd.DataFrame(rows)


def summarise_errors_by_period(errors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create and save error summaries by month, weekday and demand regime."""
    prepared = _add_period_columns(errors)
    summaries = {
        "month": _summarise_group(prepared, "month"),
        "day_of_week": _summarise_group(prepared, "day_of_week"),
        "demand_regime": _summarise_group(prepared, "demand_regime"),
        "quarter": _summarise_group(prepared, "quarter"),
        "is_weekend": _summarise_group(prepared, "is_weekend"),
    }
    ensure_dir(TABLES_DIR)
    summaries["month"].to_csv(TABLES_DIR / "error_summary_by_month.csv", index=False)
    summaries["day_of_week"].to_csv(TABLES_DIR / "error_summary_by_day_of_week.csv", index=False)
    summaries["demand_regime"].to_csv(TABLES_DIR / "error_summary_by_demand_regime.csv", index=False)
    return summaries


def analyse_incomplete_days(daily_df: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    """Analyse whether incomplete prepared days affected forecast evaluation."""
    required_columns = ["date", "coverage_ratio", "has_incomplete_day", "settlement_period_count", "expected_settlement_period_count"]
    available_columns = [column for column in required_columns if column in daily_df.columns]
    impact = errors[["date", "actual"]].merge(daily_df[available_columns], on="date", how="left")
    for forecast_column in [column for column in FORECAST_COLUMNS if column in errors.columns]:
        model = forecast_column.removesuffix("_forecast")
        impact[f"{model}_absolute_error"] = errors[f"{model}_absolute_error"]
    ensure_dir(TABLES_DIR)
    impact.to_csv(TABLES_DIR / "incomplete_day_forecast_impact.csv", index=False)
    incomplete_count = int(impact.get("has_incomplete_day", pd.Series(dtype=bool)).fillna(False).sum())
    if incomplete_count == 0:
        print("No incomplete test days were found in the forecast evaluation period.")
    else:
        print(f"Incomplete test days found: {incomplete_count}")
    return impact


def analyse_residual_autocorrelation(errors: pd.DataFrame, max_lag: int = 30) -> pd.DataFrame:
    """Calculate SARIMA/SARIMAX residual autocorrelation for lags 1 to 30."""
    rows: list[dict[str, float | int | str]] = []
    for model in ["sarima", "sarimax"]:
        error_column = f"{model}_error"
        if error_column not in errors.columns:
            continue
        series = errors[error_column].dropna()
        for lag in range(1, max_lag + 1):
            rows.append({"model": model, "lag": lag, "autocorrelation": series.autocorr(lag=lag)})
    autocorrelation = pd.DataFrame(rows)
    ensure_dir(TABLES_DIR)
    autocorrelation.to_csv(TABLES_DIR / "statistical_residual_autocorrelation.csv", index=False)
    return autocorrelation


def compare_exogenous_feature_correlations(daily_df: pd.DataFrame, target: str = DEFAULT_TARGET) -> pd.DataFrame:
    """Calculate candidate exogenous correlations with nd_mean and nd_peak-style targets."""
    target_columns = [column for column in [target, "nd_mean", "nd_peak"] if column in daily_df.columns]
    target_columns = list(dict.fromkeys(target_columns))
    rows: list[dict[str, float | str]] = []
    for feature in [column for column in EXOGENOUS_CANDIDATES if column in daily_df.columns]:
        feature_series = pd.to_numeric(daily_df[feature], errors="coerce")
        row: dict[str, float | str] = {"feature": feature}
        for target_column in target_columns:
            row[f"correlation_with_{target_column}"] = feature_series.corr(pd.to_numeric(daily_df[target_column], errors="coerce"))
        rows.append(row)
    correlations = pd.DataFrame(rows)
    ensure_dir(TABLES_DIR)
    correlations.to_csv(TABLES_DIR / "exogenous_feature_correlation_with_targets.csv", index=False)
    return correlations


def _weekly_seasonal_naive_forecast(train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """Forecast using the same weekday from the previous week."""
    history = train.dropna()
    fallback = float(history.iloc[-1])
    values: list[float] = []
    for timestamp in test_index:
        previous_week = timestamp - pd.Timedelta(days=7)
        values.append(float(history.loc[previous_week]) if previous_week in history.index else fallback)
    return pd.Series(values, index=test_index, name="weekly_seasonal_naive_forecast")


def _refined_benchmark_comparison(daily_df: pd.DataFrame, errors: pd.DataFrame, target: str) -> pd.DataFrame:
    """Compare weekly seasonal naive against year-over-year seasonal naive."""
    series = daily_df[["date", target]].dropna().sort_values("date").set_index("date")[target].asfreq("D")
    test_index = pd.DatetimeIndex(pd.to_datetime(errors["date"]))
    train = series.loc[series.index < test_index.min()].dropna()
    actual = pd.Series(errors["actual"].to_numpy(), index=test_index)
    weekly = _weekly_seasonal_naive_forecast(train, test_index)
    yearly = seasonal_naive_forecast(train, test_index)

    rows = [
        {"model": "weekly_seasonal_naive", **evaluate_forecast(actual, weekly)},
        {"model": "year_over_year_seasonal_naive", **evaluate_forecast(actual, yearly)},
    ]
    comparison = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    ensure_dir(TABLES_DIR)
    comparison.to_csv(TABLES_DIR / "refined_benchmark_comparison.csv", index=False)
    return comparison


def _plot_error_by_month(month_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot monthly MAE by model."""
    ensure_dir(output_path.parent)
    pivot = month_summary.pivot(index="month", columns="model", values="mae")
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Forecast MAE by month")
    ax.set_xlabel("month")
    ax.set_ylabel("MAE")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_error_by_demand_regime(regime_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot demand-regime MAE by model."""
    ensure_dir(output_path.parent)
    pivot = regime_summary.pivot(index="demand_regime", columns="model", values="mae")
    ax = pivot.reindex(["low", "normal", "high"]).plot(kind="bar", figsize=(10, 6))
    ax.set_title("Forecast MAE by demand regime")
    ax.set_xlabel("demand regime")
    ax.set_ylabel("MAE")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_residual_autocorrelation(autocorrelation: pd.DataFrame, output_path: Path) -> None:
    """Plot residual autocorrelation by lag."""
    ensure_dir(output_path.parent)
    plt.figure(figsize=(12, 6))
    for model, model_df in autocorrelation.groupby("model"):
        plt.plot(model_df["lag"], model_df["autocorrelation"], marker="o", label=model)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Statistical residual autocorrelation")
    plt.xlabel("lag")
    plt.ylabel("autocorrelation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_exogenous_correlations(correlations: pd.DataFrame, output_path: Path) -> None:
    """Plot exogenous feature correlations with nd_mean where available."""
    ensure_dir(output_path.parent)
    value_column = "correlation_with_nd_mean" if "correlation_with_nd_mean" in correlations.columns else correlations.columns[-1]
    plot_df = correlations[["feature", value_column]].dropna().sort_values(value_column)
    plt.figure(figsize=(10, max(5, len(plot_df) * 0.35)))
    plt.barh(plot_df["feature"], plot_df[value_column])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Exogenous feature correlation with target")
    plt.xlabel("correlation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_refined_benchmarks(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot refined benchmark MAPE comparison."""
    ensure_dir(output_path.parent)
    plt.figure(figsize=(9, 5))
    plt.bar(comparison["model"], comparison["mape"])
    plt.title("Refined benchmark MAPE comparison")
    plt.ylabel("MAPE (%)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_model_diagnostics(target: str = DEFAULT_TARGET) -> dict[str, pd.DataFrame]:
    """Run all model diagnostics and save tables and figures."""
    daily_df = load_daily_dataset()
    baseline_forecasts = load_baseline_forecasts()
    statistical_forecasts = load_statistical_forecasts()
    _ = load_statistical_comparison()

    forecasts = _merge_forecasts(baseline_forecasts, statistical_forecasts)
    errors = calculate_error_columns(forecasts)
    summaries = summarise_errors_by_period(errors)
    incomplete_impact = analyse_incomplete_days(daily_df, errors)
    autocorrelation = analyse_residual_autocorrelation(errors)
    correlations = compare_exogenous_feature_correlations(daily_df, target=target)
    refined_benchmarks = _refined_benchmark_comparison(daily_df, errors, target=target)

    _plot_error_by_month(summaries["month"], MODELLING_FIGURES_DIR / "error_by_month.png")
    _plot_error_by_demand_regime(summaries["demand_regime"], MODELLING_FIGURES_DIR / "error_by_demand_regime.png")
    _plot_residual_autocorrelation(autocorrelation, MODELLING_FIGURES_DIR / "statistical_residual_autocorrelation.png")
    _plot_exogenous_correlations(correlations, MODELLING_FIGURES_DIR / "exogenous_target_correlation.png")
    _plot_refined_benchmarks(refined_benchmarks, MODELLING_FIGURES_DIR / "refined_benchmark_comparison.png")

    return {
        "errors": errors,
        "error_summary_by_month": summaries["month"],
        "error_summary_by_day_of_week": summaries["day_of_week"],
        "error_summary_by_demand_regime": summaries["demand_regime"],
        "incomplete_day_forecast_impact": incomplete_impact,
        "statistical_residual_autocorrelation": autocorrelation,
        "exogenous_feature_correlation_with_targets": correlations,
        "refined_benchmark_comparison": refined_benchmarks,
    }


def parse_args() -> argparse.Namespace:
    """Parse diagnostics command-line arguments."""
    parser = argparse.ArgumentParser(description="Run model diagnostics for NESO demand forecasts.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Prepared daily target column, default nd_mean.")
    return parser.parse_args()


def main() -> None:
    """Run the diagnostics CLI."""
    args = parse_args()
    outputs = run_model_diagnostics(target=args.target)
    print("Model diagnostics complete.")
    for name, table in outputs.items():
        print(f"{name}: {table.shape[0]} rows, {table.shape[1]} columns")
    print(f"Saved diagnostic tables to {TABLES_DIR}")
    print(f"Saved diagnostic figures to {MODELLING_FIGURES_DIR}")


if __name__ == "__main__":
    main()
