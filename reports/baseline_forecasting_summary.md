# Baseline Forecasting Summary

## Executive Summary

This phase establishes baseline demand forecasting performance using the processed daily NESO dataset. It deliberately uses simple, transparent forecasting methods only. No SARIMAX, Prophet, machine learning, deep learning or Monte Carlo simulation models are included in this phase.

The purpose of these baselines is to define a minimum performance level that future modelling phases must improve on before their additional complexity is justified.

## Dataset Used

The default modelling input is:

`data/processed/daily_demand_2019_2025.csv`

The local data-preparation run reported 2,557 rows and 32 columns. Key columns include daily demand target candidates, calendar features, coverage/data-quality flags, renewable generation and capacity variables, and interconnector flow variables.

The processed CSV is generated locally by:

```bash
python src/prepare_data.py
```

It is not committed to Git because processed data outputs are generated artefacts.

## Target Variable

The default baseline target is:

`nd_mean`

This is recommended as the first daily baseline forecasting target because it represents average daily national demand and smooths half-hourly volatility. `nd_peak` should be retained for later capacity-pressure and scenario analysis. `tsd_mean`, `tsd_peak`, `england_wales_demand_mean` and `england_wales_demand_peak` remain useful alternative target definitions for sensitivity checks.

## Train/Test Split

The default split is chronological:

- Training data: all rows before `2025-01-01`
- Test data: `2025-01-01` to `2025-12-31`

The split is time-based and does not shuffle observations.

## Models Compared

Three baseline models are compared:

- Naive forecast: each forecast equals the last observed training value.
- Seasonal naive forecast: each forecast uses the same calendar day from the previous year where available, with a conservative fallback where it is unavailable.
- Holt-Winters Exponential Smoothing: uses additive trend and yearly seasonality where feasible. If yearly seasonality fails or is unsuitable, the code falls back to simpler exponential smoothing specifications and finally to the naive baseline.

## Metrics Used

The comparison table reports:

- MAE
- RMSE
- MAPE

Outputs are saved to:

- `outputs/tables/baseline_model_comparison.csv`
- `outputs/tables/baseline_forecasts.csv`
- `outputs/figures/modelling/actual_vs_baseline_forecasts.png`
- `outputs/figures/modelling/forecast_errors_by_model.png`

## Current Results

Codex did not generate metric results because the processed daily CSV is a local generated data file and is intentionally not committed. Running the command below locally will populate the comparison and forecast output tables:

```bash
python src/baseline_models.py --target nd_mean
```

After local execution, the best baseline should be identified primarily by lowest MAE, with RMSE and MAPE used as supporting diagnostics.

## Next Modelling Phase

The next modelling phase should review whether seasonal naive or Holt-Winters already provides a strong benchmark. Only after this baseline has been assessed should the project introduce SARIMA/SARIMAX or Prophet-style models.

Future work should still avoid scenario simulation until forecast errors, validation design and target-variable definitions have been reviewed.
