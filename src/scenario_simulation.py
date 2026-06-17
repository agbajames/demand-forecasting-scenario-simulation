"""Scenario simulation and capacity-pressure analysis for NESO demand forecasts."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .utils import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR, ensure_dir
except ImportError:  # Allows `python src/scenario_simulation.py` from the project root.
    from utils import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR, ensure_dir

DEFAULT_DAILY_DATA_PATH = PROCESSED_DATA_DIR / "daily_demand_2019_2025.csv"
DEFAULT_TARGET = "nd_mean"
DEFAULT_TEST_START = "2025-01-01"
DEFAULT_TEST_END = "2025-12-31"
DEFAULT_BASE_FORECAST_COLUMN = "gradient_boosting_forecast"
FEATURE_FORECASTS_PATH = TABLES_DIR / "feature_model_forecasts.csv"
FORECAST_DESIGN_COMPARISON_PATH = TABLES_DIR / "forecast_design_comparison.csv"
FEATURE_REGIME_COMPARISON_PATH = TABLES_DIR / "feature_model_regime_comparison.csv"
SCENARIO_FIGURES_DIR = FIGURES_DIR / "scenario_simulation"


def _project_path(path_value: str | Path) -> Path:
    """Resolve repository-relative paths against the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_daily_dataset(path: str | Path = DEFAULT_DAILY_DATA_PATH) -> pd.DataFrame:
    """Load the processed daily demand dataset."""
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


def load_feature_forecasts(path: str | Path = FEATURE_FORECASTS_PATH) -> pd.DataFrame:
    """Load feature-model forecasts from Phase 4."""
    forecast_path = _project_path(path)
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"Feature-model forecasts not found at {forecast_path}. "
            "Run `python src/feature_models.py --target nd_mean` before scenario simulation."
        )
    forecasts = pd.read_csv(forecast_path)
    if "date" not in forecasts.columns:
        raise KeyError("Feature forecast output must include a `date` column.")
    forecasts["date"] = pd.to_datetime(forecasts["date"], errors="coerce")
    if forecasts["date"].isna().any():
        raise ValueError("Feature forecast output contains unparseable dates.")
    return forecasts.sort_values("date").reset_index(drop=True)


def load_forecast_design_comparison(path: str | Path = FORECAST_DESIGN_COMPARISON_PATH) -> pd.DataFrame:
    """Load forecast-design comparison metrics where available."""
    comparison_path = _project_path(path)
    return pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()


def _load_feature_regime_comparison(path: str | Path = FEATURE_REGIME_COMPARISON_PATH) -> pd.DataFrame:
    """Load feature-model regime metrics where available."""
    regime_path = _project_path(path)
    return pd.read_csv(regime_path) if regime_path.exists() else pd.DataFrame()


def choose_base_forecast(
    forecasts: pd.DataFrame,
    base_forecast_column: str = DEFAULT_BASE_FORECAST_COLUMN,
) -> pd.DataFrame:
    """Choose the operational one-day-ahead base forecast for scenario simulation."""
    if base_forecast_column not in forecasts.columns:
        available = [column for column in forecasts.columns if column.endswith("_forecast")]
        raise KeyError(
            f"Base forecast column `{base_forecast_column}` is not available. "
            f"Available forecast columns: {available}."
        )
    output = forecasts.copy()
    output["base_forecast"] = pd.to_numeric(output[base_forecast_column], errors="coerce")
    if output["base_forecast"].isna().any():
        raise ValueError(f"Base forecast column `{base_forecast_column}` contains missing or non-numeric values.")
    return output


def estimate_residual_distribution(
    forecasts: pd.DataFrame,
    base_forecast_column: str = "base_forecast",
) -> pd.Series:
    """Estimate empirical residual noise from operational forecast errors."""
    if "actual" not in forecasts.columns:
        raise KeyError("Feature forecast output must include `actual` to estimate residual uncertainty.")
    actual = pd.to_numeric(forecasts["actual"], errors="coerce")
    base = pd.to_numeric(forecasts[base_forecast_column], errors="coerce")
    residuals = (actual - base).dropna()
    if residuals.empty:
        raise ValueError("No residuals are available to estimate simulation uncertainty.")
    centred = residuals - residuals.mean()
    return centred.rename("residual")


def _add_calendar_and_exogenous_context(
    forecasts: pd.DataFrame,
    daily: pd.DataFrame,
    target: str,
    test_start: str,
    test_end: str,
) -> pd.DataFrame:
    """Merge forecast rows with calendar and renewable context for scenario rules."""
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    daily_context = daily.loc[(daily["date"] >= start) & (daily["date"] <= end)].copy()
    daily_context["month"] = daily_context["date"].dt.month
    daily_context["day_of_week"] = daily_context["date"].dt.dayofweek
    daily_context["is_weekend"] = daily_context["day_of_week"] >= 5
    daily_context["is_winter"] = daily_context["month"].isin([12, 1, 2])
    daily_context["is_peak_season"] = daily_context["month"].isin([11, 12, 1, 2])
    context_columns = [
        column
        for column in [
            "date",
            target,
            "month",
            "day_of_week",
            "is_weekend",
            "is_winter",
            "is_peak_season",
            "embedded_wind_generation_mean",
            "embedded_solar_generation_mean",
            "demand_regime",
        ]
        if column in daily_context.columns
    ]
    output = forecasts.merge(daily_context[context_columns], on="date", how="left", suffixes=("", "_daily"))
    if "actual" not in output.columns and target in output.columns:
        output["actual"] = output[target]
    if "demand_regime" not in output.columns and "demand_regime_daily" in output.columns:
        output["demand_regime"] = output["demand_regime_daily"]
    if "is_winter" not in output.columns:
        output["is_winter"] = output["date"].dt.month.isin([12, 1, 2])
    if "is_peak_season" not in output.columns:
        output["is_peak_season"] = output["date"].dt.month.isin([11, 12, 1, 2])
    return output


def _low_renewable_mask(context: pd.DataFrame, daily: pd.DataFrame, test_start: str) -> pd.Series:
    """Identify low-renewable days using a simple embedded wind and solar proxy."""
    renewable_columns = [
        column
        for column in ["embedded_wind_generation_mean", "embedded_solar_generation_mean"]
        if column in daily.columns and column in context.columns
    ]
    if not renewable_columns:
        return pd.Series(False, index=context.index)
    train = daily.loc[daily["date"] < pd.Timestamp(test_start), renewable_columns].sum(axis=1, min_count=1)
    threshold = train.quantile(0.25)
    test_renewable = context[renewable_columns].sum(axis=1, min_count=1)
    return test_renewable <= threshold


def create_scenario_adjustments(
    forecast_context: pd.DataFrame,
    daily: pd.DataFrame,
    test_start: str = DEFAULT_TEST_START,
    high_demand_uplift: float = 0.05,
    low_renewable_uplift: float = 0.02,
    winter_peak_uplift: float = 0.07,
    combined_uplift_cap: float = 0.12,
) -> pd.DataFrame:
    """Create scenario uplift adjustments for each forecast date."""
    context = forecast_context.copy()
    high_demand_mask = context.get("demand_regime", pd.Series("", index=context.index)).eq("high")
    peak_season_mask = context.get("is_peak_season", pd.Series(False, index=context.index)).fillna(False).astype(bool)
    winter_mask = context.get("is_winter", pd.Series(False, index=context.index)).fillna(False).astype(bool)
    low_renewable_mask = _low_renewable_mask(context, daily, test_start)

    rows: list[pd.DataFrame] = []
    definitions = {
        "baseline": pd.Series(0.0, index=context.index),
        "high_demand_stress": np.where(high_demand_mask | peak_season_mask, high_demand_uplift, 0.0),
        "low_renewable_stress": np.where(low_renewable_mask, low_renewable_uplift, 0.0),
        "winter_peak_stress": np.where(winter_mask | peak_season_mask, winter_peak_uplift, 0.0),
    }
    combined = (
        pd.Series(definitions["high_demand_stress"], index=context.index)
        + pd.Series(definitions["low_renewable_stress"], index=context.index)
        + pd.Series(definitions["winter_peak_stress"], index=context.index)
    ).clip(upper=combined_uplift_cap)
    definitions["combined_stress"] = combined

    for scenario, uplift in definitions.items():
        scenario_frame = context.copy()
        scenario_frame["scenario"] = scenario
        scenario_frame["scenario_uplift_pct"] = pd.Series(uplift, index=context.index).astype(float)
        scenario_frame["adjusted_forecast"] = scenario_frame["base_forecast"] * (1 + scenario_frame["scenario_uplift_pct"])
        scenario_frame["scenario_assumption"] = _scenario_assumption_text(scenario, combined_uplift_cap)
        rows.append(scenario_frame)
    return pd.concat(rows, ignore_index=True)


def _scenario_assumption_text(scenario: str, combined_uplift_cap: float) -> str:
    """Return concise documentation for each scenario."""
    assumptions = {
        "baseline": "No systematic uplift; uncertainty comes from empirical forecast residuals.",
        "high_demand_stress": "Applies a 5% uplift on high-demand or peak-season days.",
        "low_renewable_stress": "Applies a 2% uplift on low embedded wind/solar proxy days; this is a simplified stress proxy, not a physical grid model.",
        "winter_peak_stress": "Applies a 7% uplift on winter or peak-season days.",
        "combined_stress": f"Combines high-demand, low-renewable and winter stress assumptions with total uplift capped at {combined_uplift_cap:.0%}.",
    }
    return assumptions[scenario]


def simulate_demand_paths(
    scenario_adjustments: pd.DataFrame,
    residuals: pd.Series,
    n_simulations: int = 1000,
    random_seed: int = 42,
    capacity_threshold: float | None = None,
) -> pd.DataFrame:
    """Simulate daily demand distributions for each scenario and date."""
    if n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer.")
    rng = np.random.default_rng(random_seed)
    residual_values = residuals.dropna().to_numpy(dtype=float)
    if residual_values.size == 0:
        raise ValueError("Residual distribution is empty.")

    rows: list[dict[str, object]] = []
    for _, row in scenario_adjustments.iterrows():
        noise = rng.choice(residual_values, size=n_simulations, replace=True)
        simulated = float(row["adjusted_forecast"]) + noise
        threshold = np.nan if capacity_threshold is None else float(capacity_threshold)
        exceedance_probability = float(np.mean(simulated > threshold)) if np.isfinite(threshold) else np.nan
        rows.append(
            {
                "date": row["date"],
                "scenario": row["scenario"],
                "base_forecast": float(row["base_forecast"]),
                "scenario_uplift_pct": float(row["scenario_uplift_pct"]),
                "simulated_mean": float(np.mean(simulated)),
                "simulated_p05": float(np.quantile(simulated, 0.05)),
                "simulated_p50": float(np.quantile(simulated, 0.50)),
                "simulated_p95": float(np.quantile(simulated, 0.95)),
                "capacity_threshold": threshold,
                "exceedance_probability": exceedance_probability,
                "actual": row.get("actual", np.nan),
                "demand_regime": row.get("demand_regime", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _default_capacity_threshold(daily: pd.DataFrame, target: str, test_start: str) -> tuple[float, pd.DataFrame]:
    """Calculate the default capacity threshold from training-period demand."""
    if target not in daily.columns:
        raise KeyError(f"Target `{target}` is not present in the processed daily dataset.")
    train = daily.loc[daily["date"] < pd.Timestamp(test_start), ["date", target]].dropna()
    if train.empty:
        raise ValueError("Training data is empty; cannot calculate default capacity threshold.")
    threshold = float(train[target].quantile(0.95))
    documentation = pd.DataFrame(
        [
            {
                "target": target,
                "capacity_threshold": threshold,
                "threshold_source": "95th percentile of training-period actual demand",
                "training_start": train["date"].min().date().isoformat(),
                "training_end": train["date"].max().date().isoformat(),
                "training_rows": len(train),
                "percentile": 0.95,
            }
        ]
    )
    return threshold, documentation


def calculate_capacity_pressure_metrics(daily_summary: pd.DataFrame) -> pd.DataFrame:
    """Calculate scenario-level capacity-pressure metrics."""
    rows: list[dict[str, object]] = []
    for scenario, scenario_df in daily_summary.groupby("scenario", sort=False):
        rows.append(
            {
                "scenario": scenario,
                "n_days": len(scenario_df),
                "capacity_threshold": float(scenario_df["capacity_threshold"].iloc[0]),
                "expected_exceedance_days": float(scenario_df["exceedance_probability"].sum()),
                "max_exceedance_probability": float(scenario_df["exceedance_probability"].max()),
                "mean_exceedance_probability": float(scenario_df["exceedance_probability"].mean()),
                "days_with_exceedance_probability_above_25pct": int((scenario_df["exceedance_probability"] > 0.25).sum()),
                "days_with_exceedance_probability_above_50pct": int((scenario_df["exceedance_probability"] > 0.50).sum()),
                "days_with_exceedance_probability_above_75pct": int((scenario_df["exceedance_probability"] > 0.75).sum()),
                "simulated_peak_p50": float(scenario_df["simulated_p50"].max()),
                "simulated_peak_p95": float(scenario_df["simulated_p95"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("expected_exceedance_days", ascending=False).reset_index(drop=True)


def summarise_scenarios(
    daily_summary: pd.DataFrame,
    threshold_documentation: pd.DataFrame,
    forecast_design_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact scenario-method summary for documentation."""
    base_note = "gradient_boosting_forecast from feature_model_forecasts.csv; operational one-day-ahead baseline"
    best_design = ""
    if not forecast_design_comparison.empty and {"model", "forecast_design", "mape"}.issubset(forecast_design_comparison.columns):
        best_row = forecast_design_comparison.dropna(subset=["mape"]).sort_values("mape").head(1)
        if not best_row.empty:
            best_design = f"{best_row.iloc[0]['model']} ({best_row.iloc[0]['forecast_design']}) MAPE {best_row.iloc[0]['mape']:.2f}%"
    scenario_names = ", ".join(daily_summary["scenario"].drop_duplicates().tolist())
    return pd.DataFrame(
        [
            {
                "base_forecast": base_note,
                "forecast_design_caveat": "Scenario results inherit the one-day-ahead forecast design unless a strict recursive baseline is substituted.",
                "capacity_threshold": float(threshold_documentation["capacity_threshold"].iloc[0]),
                "threshold_definition": threshold_documentation["threshold_source"].iloc[0],
                "scenarios": scenario_names,
                "forecast_validation_context": best_design,
            }
        ]
    )


def _plot_exceedance_probability(daily_summary: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(14, 7))
    for scenario, scenario_df in daily_summary.groupby("scenario", sort=False):
        plt.plot(scenario_df["date"], scenario_df["exceedance_probability"], label=scenario, linewidth=1.3)
    plt.title("Scenario capacity-threshold exceedance probability")
    plt.xlabel("date")
    plt.ylabel("exceedance probability")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_simulated_demand_ranges(daily_summary: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    scenarios = daily_summary["scenario"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, max(4, 2.6 * len(scenarios))), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        scenario_df = daily_summary.loc[daily_summary["scenario"] == scenario].sort_values("date")
        dates = pd.to_datetime(scenario_df["date"]).dt.to_pydatetime()
        ax.fill_between(
            dates,
            scenario_df["simulated_p05"].to_numpy(dtype=float),
            scenario_df["simulated_p95"].to_numpy(dtype=float),
            alpha=0.25,
            label="5th-95th percentile",
        )
        ax.plot(dates, scenario_df["simulated_p50"].to_numpy(dtype=float), label="median", linewidth=1.2)
        ax.plot(dates, scenario_df["capacity_threshold"].to_numpy(dtype=float), label="capacity threshold", linestyle="--", linewidth=1)
        ax.set_title(scenario)
        ax.set_ylabel("demand")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("date")
    fig.suptitle("Scenario simulated demand ranges", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_capacity_pressure_summary(summary: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plot_df = summary.sort_values("expected_exceedance_days", ascending=True)
    plt.figure(figsize=(11, 6))
    plt.barh(plot_df["scenario"], plot_df["expected_exceedance_days"])
    plt.title("Expected capacity-threshold exceedance days by scenario")
    plt.xlabel("expected exceedance days")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_outputs(
    daily_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    threshold_documentation: pd.DataFrame,
    method_summary: pd.DataFrame,
) -> None:
    """Save scenario simulation tables and figures."""
    ensure_dir(TABLES_DIR)
    ensure_dir(SCENARIO_FIGURES_DIR)
    daily_summary.to_csv(TABLES_DIR / "scenario_daily_simulation_summary.csv", index=False)
    scenario_summary.to_csv(TABLES_DIR / "scenario_capacity_pressure_summary.csv", index=False)
    threshold_documentation.to_csv(TABLES_DIR / "capacity_threshold_documentation.csv", index=False)
    method_summary.to_csv(TABLES_DIR / "scenario_simulation_method_summary.csv", index=False)
    _plot_exceedance_probability(daily_summary, SCENARIO_FIGURES_DIR / "scenario_exceedance_probability.png")
    _plot_simulated_demand_ranges(daily_summary, SCENARIO_FIGURES_DIR / "scenario_simulated_demand_ranges.png")
    _plot_capacity_pressure_summary(scenario_summary, SCENARIO_FIGURES_DIR / "scenario_capacity_pressure_summary.png")


def run_scenario_simulation(
    input_path: str | Path = DEFAULT_DAILY_DATA_PATH,
    target: str = DEFAULT_TARGET,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
    capacity_threshold: float | None = None,
    n_simulations: int = 1000,
    random_seed: int = 42,
    base_forecast_column: str = DEFAULT_BASE_FORECAST_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full scenario simulation and capacity-pressure workflow."""
    daily = load_daily_dataset(input_path)
    feature_forecasts = load_feature_forecasts()
    feature_forecasts = feature_forecasts.loc[
        (feature_forecasts["date"] >= pd.Timestamp(test_start))
        & (feature_forecasts["date"] <= pd.Timestamp(test_end))
    ].copy()
    if feature_forecasts.empty:
        raise ValueError("Feature forecasts are empty for the requested test period.")
    base_forecasts = choose_base_forecast(feature_forecasts, base_forecast_column=base_forecast_column)
    forecast_context = _add_calendar_and_exogenous_context(base_forecasts, daily, target, test_start, test_end)
    residuals = estimate_residual_distribution(forecast_context, base_forecast_column="base_forecast")

    if capacity_threshold is None:
        threshold, threshold_documentation = _default_capacity_threshold(daily, target, test_start)
    else:
        threshold = float(capacity_threshold)
        threshold_documentation = pd.DataFrame(
            [
                {
                    "target": target,
                    "capacity_threshold": threshold,
                    "threshold_source": "user-supplied CLI value",
                    "training_start": "",
                    "training_end": "",
                    "training_rows": "",
                    "percentile": "",
                }
            ]
        )

    scenario_adjustments = create_scenario_adjustments(forecast_context, daily, test_start=test_start)
    daily_summary = simulate_demand_paths(
        scenario_adjustments,
        residuals,
        n_simulations=n_simulations,
        random_seed=random_seed,
        capacity_threshold=threshold,
    )
    scenario_summary = calculate_capacity_pressure_metrics(daily_summary)
    method_summary = summarise_scenarios(
        daily_summary,
        threshold_documentation,
        load_forecast_design_comparison(),
    )
    _save_outputs(daily_summary, scenario_summary, threshold_documentation, method_summary)
    return daily_summary, scenario_summary, threshold_documentation


def parse_args() -> argparse.Namespace:
    """Parse scenario simulation command-line arguments."""
    parser = argparse.ArgumentParser(description="Run NESO demand scenario simulation and capacity-pressure analysis.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DAILY_DATA_PATH, help="Processed daily dataset path.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target demand column.")
    parser.add_argument("--test-start", default=DEFAULT_TEST_START, help="First simulation date.")
    parser.add_argument("--test-end", default=DEFAULT_TEST_END, help="Last simulation date.")
    parser.add_argument("--n-simulations", type=int, default=1000, help="Number of Monte Carlo paths per scenario.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for residual sampling.")
    parser.add_argument("--capacity-threshold", type=float, default=None, help="Optional fixed capacity threshold.")
    parser.add_argument(
        "--base-forecast-column",
        default=DEFAULT_BASE_FORECAST_COLUMN,
        help="Forecast column from feature_model_forecasts.csv to use as the base forecast.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the scenario simulation CLI."""
    args = parse_args()
    daily_summary, scenario_summary, threshold_documentation = run_scenario_simulation(
        input_path=args.input,
        target=args.target,
        test_start=args.test_start,
        test_end=args.test_end,
        capacity_threshold=args.capacity_threshold,
        n_simulations=args.n_simulations,
        random_seed=args.random_seed,
        base_forecast_column=args.base_forecast_column,
    )
    print("Capacity threshold documentation:")
    print(threshold_documentation)
    print("Scenario capacity-pressure summary:")
    print(scenario_summary)
    print(f"Daily scenario rows: {len(daily_summary)}")
    print(f"Saved scenario outputs to {TABLES_DIR} and {SCENARIO_FIGURES_DIR}")


if __name__ == "__main__":
    main()
