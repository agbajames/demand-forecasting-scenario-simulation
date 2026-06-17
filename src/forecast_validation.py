"""Forecast-design validation and leakage audit for feature demand models."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .feature_models import (
        DEFAULT_DAILY_DATA_PATH,
        DEFAULT_TARGET,
        DEFAULT_TEST_END,
        DEFAULT_TEST_START,
        EXOGENOUS_CANDIDATE_TERMS,
        LAGS,
        create_calendar_features,
        create_demand_regime_labels,
        create_lag_features,
        create_rolling_features,
        fit_gradient_boosting_model,
        fit_random_forest_model,
        fit_ridge_model,
        select_feature_columns,
    )
    from .utils import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR, ensure_dir
except ImportError:  # Allows `python src/forecast_validation.py` from the project root.
    from feature_models import (
        DEFAULT_DAILY_DATA_PATH,
        DEFAULT_TARGET,
        DEFAULT_TEST_END,
        DEFAULT_TEST_START,
        EXOGENOUS_CANDIDATE_TERMS,
        LAGS,
        create_calendar_features,
        create_demand_regime_labels,
        create_lag_features,
        create_rolling_features,
        fit_gradient_boosting_model,
        fit_random_forest_model,
        fit_ridge_model,
        select_feature_columns,
    )
    from utils import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR, ensure_dir

MODELLING_FIGURES_DIR = FIGURES_DIR / "modelling"
FEATURE_COMPARISON_PATH = TABLES_DIR / "feature_model_comparison.csv"
BASELINE_COMPARISON_PATH = TABLES_DIR / "baseline_model_comparison.csv"
REFINED_BENCHMARK_PATH = TABLES_DIR / "refined_benchmark_comparison.csv"

ROLLING_FEATURES = {
    "rolling_7_mean": ("mean", 7, 3),
    "rolling_14_mean": ("mean", 14, 7),
    "rolling_30_mean": ("mean", 30, 14),
    "rolling_7_max": ("max", 7, 3),
    "rolling_30_max": ("max", 30, 14),
}
CALENDAR_FEATURES = ["month", "day_of_week", "quarter", "is_weekend", "is_winter", "is_summer", "is_peak_season"]
FEATURE_MODEL_NAMES = ["ridge", "random_forest", "gradient_boosting"]


def _project_path(path_value: str | Path) -> Path:
    """Resolve repository-relative paths against the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_daily_dataset(path: str | Path = DEFAULT_DAILY_DATA_PATH) -> pd.DataFrame:
    """Load the prepared daily demand dataset for forecast validation."""
    data_path = _project_path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed daily dataset not found at {data_path}. Run `python src/prepare_data.py` locally first."
        )
    df = pd.read_csv(data_path)
    if "date" not in df.columns:
        raise KeyError("Processed daily dataset must include a `date` column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Processed daily dataset contains unparseable dates.")
    return df.sort_values("date").reset_index(drop=True)


def _metric_dict(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    """Calculate MAE, RMSE and MAPE without importing plotting-heavy modules."""
    aligned_actual, aligned_forecast = actual.align(forecast, join="inner")
    errors = aligned_actual - aligned_forecast
    non_zero_actual = aligned_actual.replace(0, np.nan)
    return {
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": float((errors.abs() / non_zero_actual).mean() * 100),
    }


def _is_target_lag(feature: str) -> bool:
    return feature.startswith("lag_") and feature.removeprefix("lag_").isdigit()


def _is_rolling_target_feature(feature: str) -> bool:
    return feature in ROLLING_FEATURES


def _is_calendar_feature(feature: str) -> bool:
    return feature in CALENDAR_FEATURES


def _is_capacity_feature(feature: str) -> bool:
    return "capacity" in feature


def _is_exogenous_feature(feature: str) -> bool:
    return any(term in feature for term in EXOGENOUS_CANDIDATE_TERMS)


def classify_feature_availability(feature: str) -> dict[str, object]:
    """Classify one feature by forecast-time availability and leakage risk."""
    if _is_target_lag(feature):
        return {
            "feature": feature,
            "feature_type": "target lag",
            "uses_target_history": True,
            "uses_same_day_observed_value": False,
            "known_at_forecast_time": "yes for one-day-ahead; recursively generated for multi-step",
            "safe_for_one_day_ahead": True,
            "safe_for_strict_multi_step": False,
            "notes": "Uses demand observed before the forecast date. Safe for next-day forecasting, but future lags must be generated recursively for a full test-period forecast.",
        }
    if _is_rolling_target_feature(feature):
        return {
            "feature": feature,
            "feature_type": "rolling target history",
            "uses_target_history": True,
            "uses_same_day_observed_value": False,
            "known_at_forecast_time": "yes for one-day-ahead; recursively generated for multi-step",
            "safe_for_one_day_ahead": True,
            "safe_for_strict_multi_step": False,
            "notes": "Calculated from shifted historical demand. It must use model forecasts, not actual future demand, in strict recursive evaluation.",
        }
    if _is_calendar_feature(feature):
        return {
            "feature": feature,
            "feature_type": "calendar",
            "uses_target_history": False,
            "uses_same_day_observed_value": False,
            "known_at_forecast_time": "yes",
            "safe_for_one_day_ahead": True,
            "safe_for_strict_multi_step": True,
            "notes": "Calendar values are known for future dates.",
        }
    if _is_capacity_feature(feature):
        return {
            "feature": feature,
            "feature_type": "capacity variable",
            "uses_target_history": False,
            "uses_same_day_observed_value": True,
            "known_at_forecast_time": "usually planned or scenario-specified",
            "safe_for_one_day_ahead": True,
            "safe_for_strict_multi_step": True,
            "notes": "Capacity is more likely to be known or planned than realised generation, but this should still be checked before operational use.",
        }
    if _is_exogenous_feature(feature):
        return {
            "feature": feature,
            "feature_type": "observed exogenous variable",
            "uses_target_history": False,
            "uses_same_day_observed_value": True,
            "known_at_forecast_time": "only if forecasted or scenario-specified",
            "safe_for_one_day_ahead": False,
            "safe_for_strict_multi_step": False,
            "notes": "Realised same-day generation or flow values are retrospective unless replaced by forecasts, schedules or scenario assumptions.",
        }
    return {
        "feature": feature,
        "feature_type": "other",
        "uses_target_history": False,
        "uses_same_day_observed_value": False,
        "known_at_forecast_time": "review required",
        "safe_for_one_day_ahead": False,
        "safe_for_strict_multi_step": False,
        "notes": "Feature was not recognised by the audit rules and should be reviewed manually.",
    }


def build_feature_audit_table(feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Build the feature availability audit table for Phase 4 feature inputs."""
    default_features = [f"lag_{lag}" for lag in LAGS] + list(ROLLING_FEATURES) + CALENDAR_FEATURES
    features = feature_columns if feature_columns is not None else default_features
    rows = [classify_feature_availability(feature) for feature in features]
    return pd.DataFrame(rows).drop_duplicates(subset=["feature"]).reset_index(drop=True)


def validate_lag_feature_design() -> pd.DataFrame:
    """Summarise why lag features are valid for one-day-ahead but not direct multi-step testing."""
    return pd.DataFrame(
        [
            {
                "forecast_design": "operational_one_day_ahead",
                "target_lag_handling": "Uses actual demand observed up to the previous day for each forecast date.",
                "interpretation": "Valid next-day operational evaluation.",
            },
            {
                "forecast_design": "strict_recursive_multi_step",
                "target_lag_handling": "Uses training actuals for the first test day, then uses model forecasts inside future lag and rolling features.",
                "interpretation": "Closer to forecasting the whole 2025 period without peeking at future actual target values.",
            },
        ]
    )


def _prepare_feature_frame(daily: pd.DataFrame, target: str, test_start: str) -> pd.DataFrame:
    """Create the same feature frame used by Phase 4 models."""
    featured = create_lag_features(daily, target=target)
    featured = create_rolling_features(featured, target=target)
    featured = create_calendar_features(featured)
    train_mask = featured["date"] < pd.Timestamp(test_start)
    featured, _ = create_demand_regime_labels(featured, target=target, train_mask=train_mask)
    return featured


def _filter_strict_feature_columns(feature_columns: list[str], strict_exog_mode: str) -> list[str]:
    """Select features allowed in strict recursive evaluation."""
    if strict_exog_mode == "actual":
        return feature_columns
    if strict_exog_mode != "drop":
        raise ValueError("strict_exog_mode must be either `drop` or `actual`.")
    return [
        feature
        for feature in feature_columns
        if not (_is_exogenous_feature(feature) and not _is_capacity_feature(feature))
    ]


def run_operational_one_day_ahead_evaluation(
    target: str = DEFAULT_TARGET,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> pd.DataFrame:
    """Document the interpretation of the existing Phase 4 feature-model evaluation."""
    comparison_path = _project_path(FEATURE_COMPARISON_PATH)
    metrics = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for model in FEATURE_MODEL_NAMES:
        metric_row = metrics.loc[metrics["model"] == model].iloc[0].to_dict() if not metrics.empty and (metrics["model"] == model).any() else {}
        rows.append(
            {
                "target": target,
                "model": model,
                "forecast_design": "operational_one_day_ahead",
                "test_start": test_start,
                "test_end": test_end,
                "uses_actual_target_history_through_previous_day": True,
                "uses_actual_target_values_on_forecast_day": False,
                "interpretation": "Valid next-day operational forecast design, not a full-year-ahead forecast from the first test date.",
                "mae": metric_row.get("mae", np.nan),
                "rmse": metric_row.get("rmse", np.nan),
                "mape": metric_row.get("mape", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _target_history_features(history: list[float]) -> dict[str, float]:
    """Create lag and shifted rolling features from a model-specific target history."""
    features: dict[str, float] = {}
    for lag in LAGS:
        features[f"lag_{lag}"] = float(history[-lag]) if len(history) >= lag else np.nan
    for feature, (method, window, min_periods) in ROLLING_FEATURES.items():
        values = pd.Series(history[-window:], dtype="float64")
        if len(values) < min_periods:
            features[feature] = np.nan
        elif method == "max":
            features[feature] = float(values.max())
        else:
            features[feature] = float(values.mean())
    return features


def _seasonal_naive_from_training(train_series: pd.Series, test_dates: pd.Series) -> pd.Series:
    """Create a year-over-year seasonal naive benchmark from training history."""
    historical = train_series.dropna()
    fallback = float(historical.iloc[-1])
    values = []
    for timestamp in pd.to_datetime(test_dates):
        previous_year = timestamp - pd.DateOffset(years=1)
        if previous_year in historical.index:
            values.append(float(historical.loc[previous_year]))
        else:
            values.append(fallback)
    return pd.Series(values, index=pd.to_datetime(test_dates), name="seasonal_naive_forecast")


def run_strict_recursive_evaluation(
    input_path: str | Path = DEFAULT_DAILY_DATA_PATH,
    target: str = DEFAULT_TARGET,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
    strict_exog_mode: str = "drop",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run recursive 2025 feature-model evaluation without future actual target leakage."""
    daily = load_daily_dataset(input_path)
    featured = _prepare_feature_frame(daily, target=target, test_start=test_start)
    all_feature_columns = select_feature_columns(featured, target=target)
    feature_columns = _filter_strict_feature_columns(all_feature_columns, strict_exog_mode)
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)

    train = featured.loc[featured["date"] < start].copy()
    test = featured.loc[(featured["date"] >= start) & (featured["date"] <= end)].copy()
    if train.empty:
        raise ValueError("Training set is empty. Choose a later test start.")
    if test.empty:
        raise ValueError("Test set is empty. Check the processed dataset and test date range.")

    train[feature_columns] = train[feature_columns].apply(pd.to_numeric, errors="coerce")
    train = train.dropna(subset=[target, *feature_columns]).copy()
    X_train = train[feature_columns]
    y_train = train[target]
    fill_values = X_train.median(numeric_only=True).to_dict()

    models = {
        "ridge": fit_ridge_model(X_train, y_train),
        "random_forest": fit_random_forest_model(X_train, y_train),
        "gradient_boosting": fit_gradient_boosting_model(X_train, y_train),
    }
    histories = {model_name: train[target].astype(float).tolist() for model_name in models}
    records: list[dict[str, object]] = []

    for _, row in test.sort_values("date").iterrows():
        record: dict[str, object] = {
            "date": row["date"],
            "actual": float(row[target]),
            "demand_regime": row.get("demand_regime"),
        }
        for model_name, model in models.items():
            feature_values = _target_history_features(histories[model_name])
            for feature in feature_columns:
                if feature in feature_values:
                    continue
                if feature in row.index:
                    feature_values[feature] = row[feature]
                else:
                    feature_values[feature] = np.nan
            X_one = pd.DataFrame([feature_values], columns=feature_columns).apply(pd.to_numeric, errors="coerce")
            X_one = X_one.fillna(fill_values)
            forecast = float(model.predict(X_one)[0])
            record[f"{model_name}_recursive_forecast"] = forecast
            histories[model_name].append(forecast)
        records.append(record)

    forecasts = pd.DataFrame(records)
    comparison_rows = []
    actual_series = pd.Series(forecasts["actual"].to_numpy(), index=pd.to_datetime(forecasts["date"]))
    for model_name in FEATURE_MODEL_NAMES:
        forecast_series = pd.Series(
            forecasts[f"{model_name}_recursive_forecast"].to_numpy(),
            index=pd.to_datetime(forecasts["date"]),
        )
        comparison_rows.append(
            {
                "model": model_name,
                "forecast_design": f"strict_recursive_{strict_exog_mode}_exog",
                **_metric_dict(actual_series, forecast_series),
            }
        )

    train_series = train.set_index("date")[target].sort_index()
    seasonal_forecast = _seasonal_naive_from_training(train_series, forecasts["date"])
    comparison_rows.append(
        {
            "model": "seasonal_naive",
            "forecast_design": "benchmark",
            **_metric_dict(actual_series, seasonal_forecast),
        }
    )
    comparison = pd.DataFrame(comparison_rows)
    regime_comparison = evaluate_by_demand_regime_recursive(forecasts)
    return comparison, regime_comparison, forecasts


def evaluate_by_demand_regime_recursive(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Calculate recursive forecast metrics by demand regime."""
    rows: list[dict[str, object]] = []
    for regime, regime_df in forecasts.groupby("demand_regime", dropna=False):
        actual = pd.Series(regime_df["actual"].to_numpy(), index=pd.to_datetime(regime_df["date"]))
        for model_name in FEATURE_MODEL_NAMES:
            column = f"{model_name}_recursive_forecast"
            forecast = pd.Series(regime_df[column].to_numpy(), index=pd.to_datetime(regime_df["date"]))
            rows.append({"demand_regime": regime, "model": model_name, "rows": len(regime_df), **_metric_dict(actual, forecast)})
    return pd.DataFrame(rows)


def _load_existing_benchmark_metrics() -> pd.DataFrame:
    """Load saved seasonal naive benchmark rows if previous outputs are available."""
    rows = []
    for path in [REFINED_BENCHMARK_PATH, BASELINE_COMPARISON_PATH]:
        csv_path = _project_path(path)
        if not csv_path.exists():
            continue
        table = pd.read_csv(csv_path)
        for model_name in ["year_over_year_seasonal_naive", "seasonal_naive"]:
            if "model" in table.columns and (table["model"] == model_name).any():
                row = table.loc[table["model"] == model_name].iloc[0].to_dict()
                rows.append(
                    {
                        "model": "seasonal_naive",
                        "forecast_design": "benchmark",
                        "mae": row.get("mae", np.nan),
                        "rmse": row.get("rmse", np.nan),
                        "mape": row.get("mape", np.nan),
                    }
                )
                break
    return pd.DataFrame(rows).drop_duplicates(subset=["model"], keep="first")


def compare_forecast_designs(
    operational_summary: pd.DataFrame,
    strict_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Combine operational one-day-ahead, strict recursive and benchmark metrics."""
    operational_rows = operational_summary[["model", "forecast_design", "mae", "rmse", "mape"]].copy()
    benchmark = _load_existing_benchmark_metrics()
    frames = [operational_rows, strict_comparison]
    if not benchmark.empty:
        frames.append(benchmark)
    comparison = pd.concat(frames, ignore_index=True)
    return comparison.drop_duplicates(subset=["model", "forecast_design"], keep="first").reset_index(drop=True)


def _plot_forecast_design_mape(comparison: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plot_df = comparison.dropna(subset=["mape"]).copy()
    plot_df["label"] = plot_df["model"] + "\n" + plot_df["forecast_design"]
    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["label"], plot_df["mape"])
    plt.title("Forecast design MAPE comparison")
    plt.ylabel("MAPE (%)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_strict_recursive_forecasts(forecasts: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(14, 6))
    plt.plot(forecasts["date"], forecasts["actual"], label="actual", linewidth=2)
    for column in [f"{model}_recursive_forecast" for model in FEATURE_MODEL_NAMES]:
        plt.plot(forecasts["date"], forecasts[column], label=column, linewidth=1)
    plt.title("Strict recursive feature forecasts")
    plt.xlabel("date")
    plt.ylabel("demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_strict_recursive_regime_mape(regime_comparison: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    pivot = regime_comparison.pivot(index="demand_regime", columns="model", values="mape")
    ax = pivot.reindex(["low", "normal", "high"]).plot(kind="bar", figsize=(12, 6))
    ax.set_title("Strict recursive MAPE by demand regime")
    ax.set_xlabel("demand regime")
    ax.set_ylabel("MAPE (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_forecast_validation_outputs(
    feature_audit: pd.DataFrame,
    operational_summary: pd.DataFrame,
    strict_forecasts: pd.DataFrame,
    strict_comparison: pd.DataFrame,
    regime_comparison: pd.DataFrame,
    design_comparison: pd.DataFrame,
) -> None:
    """Save forecast-design validation tables and figures."""
    ensure_dir(TABLES_DIR)
    ensure_dir(MODELLING_FIGURES_DIR)
    feature_audit.to_csv(TABLES_DIR / "feature_availability_audit.csv", index=False)
    operational_summary.to_csv(TABLES_DIR / "operational_forecast_design_summary.csv", index=False)
    strict_forecasts.to_csv(TABLES_DIR / "strict_recursive_feature_forecasts.csv", index=False)
    strict_comparison.to_csv(TABLES_DIR / "strict_recursive_feature_comparison.csv", index=False)
    regime_comparison.to_csv(TABLES_DIR / "strict_recursive_regime_comparison.csv", index=False)
    design_comparison.to_csv(TABLES_DIR / "forecast_design_comparison.csv", index=False)
    _plot_forecast_design_mape(design_comparison, MODELLING_FIGURES_DIR / "forecast_design_mape_comparison.png")
    _plot_strict_recursive_forecasts(strict_forecasts, MODELLING_FIGURES_DIR / "strict_recursive_actual_vs_forecast.png")
    _plot_strict_recursive_regime_mape(
        regime_comparison,
        MODELLING_FIGURES_DIR / "strict_recursive_regime_mape_comparison.png",
    )


def run_forecast_validation(
    input_path: str | Path = DEFAULT_DAILY_DATA_PATH,
    target: str = DEFAULT_TARGET,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
    strict_exog_mode: str = "drop",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full forecast-design validation workflow."""
    daily = load_daily_dataset(input_path)
    featured = _prepare_feature_frame(daily, target=target, test_start=test_start)
    feature_columns = select_feature_columns(featured, target=target)
    strict_feature_columns = _filter_strict_feature_columns(feature_columns, strict_exog_mode)
    feature_audit = build_feature_audit_table(feature_columns)
    operational_summary = run_operational_one_day_ahead_evaluation(
        target=target,
        test_start=test_start,
        test_end=test_end,
    )
    strict_comparison, regime_comparison, strict_forecasts = run_strict_recursive_evaluation(
        input_path=input_path,
        target=target,
        test_start=test_start,
        test_end=test_end,
        strict_exog_mode=strict_exog_mode,
    )
    design_comparison = compare_forecast_designs(operational_summary, strict_comparison)
    feature_audit["included_in_strict_recursive_run"] = feature_audit["feature"].isin(strict_feature_columns)
    save_forecast_validation_outputs(
        feature_audit,
        operational_summary,
        strict_forecasts,
        strict_comparison,
        regime_comparison,
        design_comparison,
    )
    return feature_audit, design_comparison, regime_comparison, strict_forecasts


def parse_args() -> argparse.Namespace:
    """Parse forecast-validation command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate forecast design and feature availability.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DAILY_DATA_PATH, help="Processed daily dataset path.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column to forecast.")
    parser.add_argument("--test-start", default=DEFAULT_TEST_START, help="First date in the test set.")
    parser.add_argument("--test-end", default=DEFAULT_TEST_END, help="Last date in the test set.")
    parser.add_argument(
        "--strict-exog-mode",
        choices=["drop", "actual"],
        default="drop",
        help="How strict recursive evaluation handles observed exogenous variables.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the forecast-validation CLI."""
    args = parse_args()
    feature_audit, design_comparison, regime_comparison, strict_forecasts = run_forecast_validation(
        input_path=args.input,
        target=args.target,
        test_start=args.test_start,
        test_end=args.test_end,
        strict_exog_mode=args.strict_exog_mode,
    )
    print("Feature availability audit:")
    print(feature_audit)
    print("Forecast design comparison:")
    print(design_comparison)
    print("Strict recursive regime comparison:")
    print(regime_comparison)
    print(f"Strict recursive forecast rows: {len(strict_forecasts)}")
    print(f"Saved forecast-validation outputs to {TABLES_DIR} and {MODELLING_FIGURES_DIR}")


if __name__ == "__main__":
    main()
