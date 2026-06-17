"""Feature-engineered demand forecasting models for daily NESO demand."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .baseline_models import evaluate_forecast as baseline_evaluate_forecast
    from .utils import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR, ensure_dir
except ImportError:  # Allows `python src/feature_models.py` from the project root.
    from baseline_models import evaluate_forecast as baseline_evaluate_forecast
    from utils import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR, ensure_dir

DEFAULT_DAILY_DATA_PATH = PROCESSED_DATA_DIR / "daily_demand_2019_2025.csv"
DEFAULT_TARGET = "nd_mean"
DEFAULT_TEST_START = "2025-01-01"
DEFAULT_TEST_END = "2025-12-31"
BASELINE_COMPARISON_PATH = TABLES_DIR / "baseline_model_comparison.csv"
STATISTICAL_COMPARISON_PATH = TABLES_DIR / "statistical_model_comparison.csv"
BASELINE_FORECASTS_PATH = TABLES_DIR / "baseline_forecasts.csv"
STATISTICAL_FORECASTS_PATH = TABLES_DIR / "statistical_forecasts.csv"
REFINED_BENCHMARK_PATH = TABLES_DIR / "refined_benchmark_comparison.csv"
MODELLING_FIGURES_DIR = FIGURES_DIR / "modelling"
LAGS = [1, 2, 3, 7, 14, 28, 365]
ROLLING_WINDOWS = [7, 14, 30]
EXOGENOUS_CANDIDATE_TERMS = [
    "embedded_wind",
    "embedded_solar",
    "pump_storage",
    "pumped",
    "interconnector",
    "ifa",
    "ifa2",
    "britned",
    "moyle",
    "east_west",
    "nemo",
    "nsl",
    "eleclink",
    "viking",
]


def _project_path(path_value: str | Path) -> Path:
    """Resolve repository-relative paths against the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_optional_csv(path: str | Path) -> pd.DataFrame:
    """Read a generated CSV if available, otherwise return an empty dataframe."""
    csv_path = _project_path(path)
    return pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()


def load_daily_dataset(path: str | Path = DEFAULT_DAILY_DATA_PATH) -> pd.DataFrame:
    """Load the prepared daily dataset."""
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


def create_lag_features(df: pd.DataFrame, target: str = DEFAULT_TARGET) -> pd.DataFrame:
    """Create target lag features using only previous observations."""
    output = df.copy()
    for lag in LAGS:
        output[f"lag_{lag}"] = output[target].shift(lag)
    return output


def create_rolling_features(df: pd.DataFrame, target: str = DEFAULT_TARGET) -> pd.DataFrame:
    """Create shifted rolling features so the forecast day target is never included."""
    output = df.copy()
    shifted_target = output[target].shift(1)
    output["rolling_7_mean"] = shifted_target.rolling(7, min_periods=3).mean()
    output["rolling_14_mean"] = shifted_target.rolling(14, min_periods=7).mean()
    output["rolling_30_mean"] = shifted_target.rolling(30, min_periods=14).mean()
    output["rolling_7_max"] = shifted_target.rolling(7, min_periods=3).max()
    output["rolling_30_max"] = shifted_target.rolling(30, min_periods=14).max()
    return output


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and seasonal indicator features."""
    output = df.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    if "month" not in output.columns:
        output["month"] = output["date"].dt.month
    if "day_of_week" not in output.columns:
        output["day_of_week"] = output["date"].dt.dayofweek
    if "quarter" not in output.columns:
        output["quarter"] = output["date"].dt.quarter
    if "is_weekend" not in output.columns:
        output["is_weekend"] = output["day_of_week"] >= 5
    output["is_winter"] = output["month"].isin([12, 1, 2])
    output["is_summer"] = output["month"].isin([6, 7, 8])
    output["is_peak_season"] = output["month"].isin([11, 12, 1, 2])
    return output


def create_demand_regime_labels(
    df: pd.DataFrame,
    target: str,
    train_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Label demand regimes using training-period target thresholds only."""
    output = df.copy()
    q25, q75 = output.loc[train_mask, target].quantile([0.25, 0.75])
    output["demand_regime"] = np.select(
        [output[target] < q25, output[target] > q75],
        ["low", "high"],
        default="normal",
    )
    return output, {"low_threshold": float(q25), "high_threshold": float(q75)}


def select_feature_columns(df: pd.DataFrame, target: str = DEFAULT_TARGET) -> list[str]:
    """Select engineered, calendar and available exogenous feature columns."""
    engineered = [f"lag_{lag}" for lag in LAGS] + [
        "rolling_7_mean",
        "rolling_14_mean",
        "rolling_30_mean",
        "rolling_7_max",
        "rolling_30_max",
        "month",
        "day_of_week",
        "quarter",
        "is_weekend",
        "is_winter",
        "is_summer",
        "is_peak_season",
    ]
    exogenous = [
        column
        for column in df.columns
        if column != target and column.endswith("_mean") and any(term in column for term in EXOGENOUS_CANDIDATE_TERMS)
    ]
    columns = [column for column in [*engineered, *exogenous] if column in df.columns]
    return list(dict.fromkeys(columns))


def train_test_split_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str = DEFAULT_TARGET,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Create chronological train/test matrices without shuffling."""
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    working = df.copy()
    working[feature_columns] = working[feature_columns].apply(pd.to_numeric, errors="coerce")
    working = working.dropna(subset=[target, *feature_columns]).copy()
    train_mask = working["date"] < start
    test_mask = (working["date"] >= start) & (working["date"] <= end)
    if not train_mask.any():
        raise ValueError("Training feature set is empty. Choose a later test start or inspect lag feature availability.")
    if not test_mask.any():
        raise ValueError("Test feature set is empty. Check the processed dataset and test date range.")
    X_train = working.loc[train_mask, feature_columns]
    X_test = working.loc[test_mask, feature_columns]
    y_train = working.loc[train_mask, target]
    y_test = working.loc[test_mask, target]
    test_dates = working.loc[test_mask, "date"]
    test_metadata = working.loc[test_mask, ["date", target, "demand_regime"]].copy()
    return X_train, X_test, y_train, y_test, test_dates, test_metadata


def fit_ridge_model(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Pipeline:
    """Fit a standardised Ridge regression model."""
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    model.fit(X_train, y_train)
    return model


def fit_random_forest_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Fit a compact random forest model with deterministic settings."""
    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def fit_gradient_boosting_model(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
    """Fit a compact gradient boosting model with deterministic settings."""
    model = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_forecast(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    """Calculate MAE, RMSE and MAPE."""
    return baseline_evaluate_forecast(actual, forecast)


def evaluate_by_demand_regime(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Calculate forecast metrics by demand regime."""
    rows: list[dict[str, float | str | int]] = []
    forecast_columns = [column for column in forecasts.columns if column.endswith("_forecast")]
    for regime, regime_df in forecasts.groupby("demand_regime", dropna=False):
        actual = pd.Series(regime_df["actual"].to_numpy(), index=regime_df["date"])
        for forecast_column in forecast_columns:
            model = forecast_column.removesuffix("_forecast")
            metrics = evaluate_forecast(actual, pd.Series(regime_df[forecast_column].to_numpy(), index=regime_df["date"]))
            rows.append({"demand_regime": regime, "model": model, "rows": len(regime_df), **metrics})
    return pd.DataFrame(rows)


def _load_existing_comparisons() -> pd.DataFrame:
    """Load previous baseline/statistical comparisons if available."""
    frames = []
    for path in [BASELINE_COMPARISON_PATH, STATISTICAL_COMPARISON_PATH, REFINED_BENCHMARK_PATH]:
        csv_path = _project_path(path)
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path))
    if not frames:
        return pd.DataFrame(columns=["model", "mae", "rmse", "mape"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["model"], keep="last")


def _load_existing_forecasts() -> pd.DataFrame:
    """Load previous forecasts for benchmark columns if available."""
    baseline_path = _project_path(BASELINE_FORECASTS_PATH)
    statistical_path = _project_path(STATISTICAL_FORECASTS_PATH)
    forecasts = pd.DataFrame()
    if statistical_path.exists():
        forecasts = pd.read_csv(statistical_path)
    elif baseline_path.exists():
        forecasts = pd.read_csv(baseline_path)
    if not forecasts.empty:
        forecasts["date"] = pd.to_datetime(forecasts["date"], errors="coerce")
        keep_columns = [
            column
            for column in ["date", "seasonal_naive_forecast", "sarimax_forecast"]
            if column in forecasts.columns
        ]
        forecasts = forecasts[keep_columns]
    return forecasts


def _save_feature_importance(
    feature_columns: list[str],
    random_forest: RandomForestRegressor,
    gradient_boosting: GradientBoostingRegressor,
    ridge_model: Pipeline,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save feature importance and Ridge coefficient tables."""
    importance_rows = []
    for model_name, importances in [
        ("random_forest", random_forest.feature_importances_),
        ("gradient_boosting", gradient_boosting.feature_importances_),
    ]:
        for feature, importance in zip(feature_columns, importances):
            importance_rows.append({"model": model_name, "feature": feature, "importance": importance})
    importance = pd.DataFrame(importance_rows).sort_values(["model", "importance"], ascending=[True, False])
    ridge = ridge_model.named_steps["ridge"]
    coefficients = pd.DataFrame({"feature": feature_columns, "coefficient": ridge.coef_}).sort_values(
        "coefficient", key=lambda values: values.abs(), ascending=False
    )
    ensure_dir(TABLES_DIR)
    importance.to_csv(TABLES_DIR / "feature_model_importance.csv", index=False)
    coefficients.to_csv(TABLES_DIR / "ridge_feature_coefficients.csv", index=False)
    return importance, coefficients


def _plot_actual_vs_feature_forecasts(forecasts: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(14, 6))
    plt.plot(forecasts["date"], forecasts["actual"], label="actual", linewidth=2)
    for column in [
        "seasonal_naive_forecast",
        "sarimax_forecast",
        "ridge_forecast",
        "random_forest_forecast",
        "gradient_boosting_forecast",
    ]:
        if column in forecasts.columns:
            plt.plot(forecasts["date"], forecasts[column], label=column, linewidth=1)
    plt.title("Actual demand versus feature model forecasts")
    plt.xlabel("date")
    plt.ylabel("demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_mape_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plot_df = comparison.dropna(subset=["mape"]).sort_values("mape")
    plt.figure(figsize=(11, 6))
    plt.bar(plot_df["model"], plot_df["mape"])
    plt.title("Feature model MAPE comparison")
    plt.ylabel("MAPE (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_regime_mape(regime_comparison: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    pivot = regime_comparison.pivot(index="demand_regime", columns="model", values="mape")
    ax = pivot.reindex(["low", "normal", "high"]).plot(kind="bar", figsize=(12, 6))
    ax.set_title("Feature model MAPE by demand regime")
    ax.set_xlabel("demand regime")
    ax.set_ylabel("MAPE (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_feature_importance(importance: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    top = (
        importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(20)
        .sort_values("importance")
    )
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top 20 feature importances")
    plt.xlabel("mean tree-model importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_feature_modelling(
    input_path: str | Path = DEFAULT_DAILY_DATA_PATH,
    target: str = DEFAULT_TARGET,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run feature-engineered models and save comparisons, forecasts and figures."""
    daily = load_daily_dataset(input_path)
    daily = create_calendar_features(create_rolling_features(create_lag_features(daily, target=target), target=target))
    train_mask = daily["date"] < pd.Timestamp(test_start)
    daily, _ = create_demand_regime_labels(daily, target=target, train_mask=train_mask)
    feature_columns = select_feature_columns(daily, target=target)
    X_train, X_test, y_train, y_test, test_dates, test_metadata = train_test_split_features(
        daily, feature_columns, target=target, test_start=test_start, test_end=test_end
    )

    ridge = fit_ridge_model(X_train, y_train)
    random_forest = fit_random_forest_model(X_train, y_train)
    gradient_boosting = fit_gradient_boosting_model(X_train, y_train)

    forecasts = pd.DataFrame(
        {
            "date": test_dates.to_numpy(),
            "actual": y_test.to_numpy(),
            "ridge_forecast": ridge.predict(X_test),
            "random_forest_forecast": random_forest.predict(X_test),
            "gradient_boosting_forecast": gradient_boosting.predict(X_test),
            "demand_regime": test_metadata["demand_regime"].to_numpy(),
        }
    )
    existing_forecasts = _load_existing_forecasts()
    if not existing_forecasts.empty:
        forecasts = forecasts.merge(existing_forecasts, on="date", how="left")

    feature_rows = []
    for model, column in [
        ("ridge", "ridge_forecast"),
        ("random_forest", "random_forest_forecast"),
        ("gradient_boosting", "gradient_boosting_forecast"),
    ]:
        metrics = evaluate_forecast(pd.Series(forecasts["actual"].to_numpy(), index=forecasts["date"]), pd.Series(forecasts[column].to_numpy(), index=forecasts["date"]))
        feature_rows.append({"model": model, **metrics})

    previous = _load_existing_comparisons()
    comparison = pd.concat([previous, pd.DataFrame(feature_rows)], ignore_index=True)
    comparison = comparison.drop_duplicates(subset=["model"], keep="last").sort_values("mae").reset_index(drop=True)
    regime_comparison = evaluate_by_demand_regime(forecasts)
    importance, _ = _save_feature_importance(feature_columns, random_forest, gradient_boosting, ridge)

    ensure_dir(TABLES_DIR)
    comparison.to_csv(TABLES_DIR / "feature_model_comparison.csv", index=False)
    regime_comparison.to_csv(TABLES_DIR / "feature_model_regime_comparison.csv", index=False)
    forecasts.to_csv(TABLES_DIR / "feature_model_forecasts.csv", index=False)
    _plot_actual_vs_feature_forecasts(forecasts, MODELLING_FIGURES_DIR / "actual_vs_feature_model_forecasts.png")
    _plot_mape_comparison(comparison, MODELLING_FIGURES_DIR / "feature_model_mape_comparison.png")
    _plot_regime_mape(regime_comparison, MODELLING_FIGURES_DIR / "feature_model_regime_mape_comparison.png")
    _plot_feature_importance(importance, MODELLING_FIGURES_DIR / "feature_importance_top20.png")
    return comparison, regime_comparison, forecasts


def parse_args() -> argparse.Namespace:
    """Parse feature modelling command-line arguments."""
    parser = argparse.ArgumentParser(description="Run feature-engineered demand forecasting models.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DAILY_DATA_PATH, help="Processed daily dataset path.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column to forecast.")
    parser.add_argument("--test-start", default=DEFAULT_TEST_START, help="First date in the test set.")
    parser.add_argument("--test-end", default=DEFAULT_TEST_END, help="Last date in the test set.")
    return parser.parse_args()


def main() -> None:
    """Run the feature modelling CLI."""
    args = parse_args()
    comparison, regime_comparison, forecasts = run_feature_modelling(
        input_path=args.input,
        target=args.target,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    print("Feature model comparison:")
    print(comparison)
    print("Feature model regime comparison:")
    print(regime_comparison)
    print(f"Forecast rows: {len(forecasts)}")
    print(f"Saved feature modelling outputs to {TABLES_DIR} and {MODELLING_FIGURES_DIR}")


if __name__ == "__main__":
    main()
