# Statistical Forecasting Summary

## Executive Summary

This phase introduces practical statistical forecasting models for the prepared NESO daily demand dataset. The models are SARIMA and SARIMAX, evaluated against the Phase 2 baseline results on the same 2025 test period.

The seasonal naive model is the benchmark to beat. SARIMA or SARIMAX should only be described as an improvement if their saved metrics beat seasonal naive on the same target and test period.

No Prophet, deep learning or Monte Carlo simulation models are included in this phase.

## Dataset Used

The default modelling input is:

`data/processed/daily_demand_2019_2025.csv`

The default target variable is:

`nd_mean`

The default test period is:

- Training: all observations before `2025-01-01`
- Test: `2025-01-01` to `2025-12-31`

## Baseline Benchmark

The current local baseline result to beat is:

- Model: `seasonal_naive`
- Target: `nd_mean`
- Test period: `2025-01-01` to `2025-12-31`
- MAE: 2076.58
- RMSE: 2648.82
- MAPE: 7.93%

The statistical pipeline loads `outputs/tables/baseline_model_comparison.csv` and `outputs/tables/baseline_forecasts.csv` where available, then saves a combined comparison to:

`outputs/tables/statistical_model_comparison.csv`

## SARIMA Configuration

The initial SARIMA configuration is:

- `order=(1, 1, 1)`
- `seasonal_order=(1, 1, 1, 7)`

This is a practical first statistical model for daily data. Weekly seasonality is modelled first because it is operationally meaningful and computationally manageable. A yearly seasonal period of 365 is deliberately not attempted at this stage because it can be slow and unstable.

## SARIMAX Configuration

SARIMAX uses the same ARIMA and weekly seasonal structure as SARIMA, with selected exogenous features where present in the processed daily dataset.

Candidate exogenous variables are:

- `embedded_wind_generation_mean`
- `embedded_solar_generation_mean`
- `pump_storage_pumping_mean`
- `ifa_flow_mean`
- `ifa2_flow_mean`
- `britned_flow_mean`
- `moyle_flow_mean`
- `east_west_flow_mean`
- `nemo_flow_mean`
- `nsl_flow_mean`
- `eleclink_flow_mean`
- `viking_flow_mean`
- `is_weekend`
- `month`
- `day_of_week`

The script uses only columns that exist, fills missing exogenous values with forward/backward fill and zero fallback, and aligns train/test exogenous rows to the target dates. Exogenous features are standardised using training-period statistics and then applied to the test period. It does not use future target values.

## Outputs

Running:

```bash
python src/statistical_models.py --target nd_mean
```

saves:

- `outputs/tables/statistical_model_comparison.csv`
- `outputs/tables/statistical_forecasts.csv`
- `outputs/figures/modelling/actual_vs_statistical_forecasts.png`
- `outputs/figures/modelling/statistical_forecast_errors.png`
- `outputs/figures/modelling/model_mape_comparison.png`

## Interpretation

After local execution, compare SARIMA and SARIMAX directly with `seasonal_naive`. If neither statistical model beats seasonal naive on MAE, RMSE or MAPE, seasonal naive remains the best model so far. If one statistical model improves a metric but worsens others, the trade-off should be documented rather than described as a clear overall win.

## Next Recommended Phase

If SARIMA or SARIMAX beats the seasonal naive benchmark, the next phase can refine statistical model specification and validation. If they do not, the next phase should diagnose why the seasonal naive baseline is strong and consider feature engineering, alternative target definitions or additional external regressors before adding Prophet or simulation.

Prophet and scenario simulation should remain out of scope until the statistical comparison is reviewed.
