# Model Diagnostics and Feature Refinement Summary

## Executive Summary

Phase 3 showed that SARIMA and SARIMAX did not beat the seasonal naive benchmark for `nd_mean` on the 2025 test period. This is an important modelling result rather than a failure: the project now has a clear benchmark and a reason to inspect error structure before adding more complex methods.

The current best model remains:

- Model: `seasonal_naive`
- MAE: 2076.58
- RMSE: 2648.82
- MAPE: 7.93%

SARIMAX is weaker than seasonal naive on the current comparison. The diagnostics phase is designed to explain where and why, not to hide weak results.

## Diagnostic Inputs

The diagnostics module uses:

- `data/processed/daily_demand_2019_2025.csv`
- `outputs/tables/baseline_forecasts.csv`
- `outputs/tables/statistical_forecasts.csv`
- `outputs/tables/statistical_model_comparison.csv`

These are generated locally and are not committed to Git.

## Error Clustering

The diagnostics pipeline summarises forecast errors by:

- month
- quarter
- day of week
- weekend versus weekday
- demand regime

Demand regimes are defined from actual 2025 target values:

- low demand: bottom 25%
- normal demand: middle 50%
- high demand: top 25%

This regime view is especially important because later capacity-pressure analysis will care more about high-demand days than average days.

## Incomplete Day Analysis

The diagnostics pipeline checks whether 2025 forecast dates have incomplete settlement-period coverage using:

- `coverage_ratio`
- `has_incomplete_day`
- `settlement_period_count`
- `expected_settlement_period_count`

The output is saved to:

`outputs/tables/incomplete_day_forecast_impact.csv`

If no incomplete test days are present, the script prints this clearly.

## Residual Autocorrelation

SARIMA and SARIMAX errors are checked for autocorrelation at lags 1 to 30. Remaining autocorrelation would indicate that the statistical models have not fully captured repeatable structure in the daily series.

Outputs:

- `outputs/tables/statistical_residual_autocorrelation.csv`
- `outputs/figures/modelling/statistical_residual_autocorrelation.png`

## Exogenous Feature Usefulness

The diagnostics pipeline calculates correlations between candidate SARIMAX exogenous variables and:

- `nd_mean`
- `nd_peak`

The output is saved to:

`outputs/tables/exogenous_feature_correlation_with_targets.csv`

This helps identify which renewable, capacity, pumping or interconnector variables may be useful and which may be adding noise.

## Refined Benchmark Comparison

The current seasonal naive benchmark is strong. Phase 3B therefore compares:

- weekly seasonal naive: same day from the previous week
- year-over-year seasonal naive: same calendar day from the previous year

The output is saved to:

`outputs/tables/refined_benchmark_comparison.csv`

This helps determine whether weekly or annual seasonal structure is driving the strong baseline performance.

## Recommended Next Modelling Step

Do not add Prophet or simulation yet. The next modelling step should first review:

- which periods and demand regimes drive SARIMA/SARIMAX errors;
- whether SARIMA/SARIMAX residuals retain strong autocorrelation;
- whether exogenous variables have useful relationships with `nd_mean` or `nd_peak`;
- whether weekly or year-over-year seasonal naive is the stronger benchmark.

If diagnostics show that exogenous variables are weak or noisy, the SARIMAX feature set should be refined before trying another model family. If residual autocorrelation remains strong, SARIMA/SARIMAX specification should be revisited with a careful, limited search rather than a large expensive grid.

Seasonal naive remains the best model so far unless future saved metrics prove otherwise.
