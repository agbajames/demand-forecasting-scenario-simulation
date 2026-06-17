# Feature Modelling Summary

## Executive Summary

Phase 4 tests whether feature-engineered models can improve on the year-over-year seasonal naive benchmark for daily `nd_mean` demand forecasting, especially on high-demand days.

The current benchmark remains:

- Model: `year_over_year_seasonal_naive`
- MAE: 2076.58
- RMSE: 2648.82
- MAPE: 7.93%

Feature models should only be described as an improvement if their saved metrics beat this benchmark on the same 2025 test period. If they do not, the result should be reported plainly.

## Dataset and Target

The default input dataset is:

`data/processed/daily_demand_2019_2025.csv`

The default target is:

`nd_mean`

The default test period is:

- Training: all rows before `2025-01-01`
- Test: `2025-01-01` to `2025-12-31`

## Features Engineered

The feature modelling pipeline creates:

- target lags: `lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_14`, `lag_28`, `lag_365`
- shifted rolling means: `rolling_7_mean`, `rolling_14_mean`, `rolling_30_mean`
- shifted rolling maxima: `rolling_7_max`, `rolling_30_max`
- calendar features: `month`, `day_of_week`, `quarter`, `is_weekend`
- seasonal flags: `is_winter`, `is_summer`, `is_peak_season`
- available renewable, capacity, pumping and interconnector mean columns

Rolling features are shifted by one day before calculation to avoid using the forecast day's actual target value.

## Demand Regime Definition

Demand regimes are defined using training-period target quantiles only:

- low demand: below the 25th percentile
- normal demand: between the 25th and 75th percentiles
- high demand: above the 75th percentile

The same thresholds are then applied to the 2025 test period. Test-period quantiles are not used to define regimes.

## Models Trained

The feature modelling phase compares:

- Ridge regression with `StandardScaler`
- Random Forest Regressor
- Gradient Boosting Regressor

These are deliberately modest, fixed-configuration models intended to test whether engineered features improve the benchmark before any further model family is introduced.

## Outputs

Running:

```bash
python src/feature_models.py --target nd_mean
```

saves:

- `outputs/tables/feature_model_comparison.csv`
- `outputs/tables/feature_model_regime_comparison.csv`
- `outputs/tables/feature_model_forecasts.csv`
- `outputs/tables/feature_model_importance.csv`
- `outputs/tables/ridge_feature_coefficients.csv`
- `outputs/figures/modelling/actual_vs_feature_model_forecasts.png`
- `outputs/figures/modelling/feature_model_mape_comparison.png`
- `outputs/figures/modelling/feature_model_regime_mape_comparison.png`
- `outputs/figures/modelling/feature_importance_top20.png`

## Current Results

Codex did not generate results because local processed data and previous forecast outputs are generated artefacts and are intentionally not committed. After local execution, compare feature models with the year-over-year seasonal naive benchmark overall and within the high-demand regime.

## Interpretation Guidance

If no feature model beats seasonal naive overall, seasonal naive remains the best model so far. If a feature model improves high-demand performance but worsens overall performance, document that trade-off rather than presenting it as a general improvement.

Feature importance should be reviewed to determine whether lag features, rolling maxima, seasonal flags or exogenous variables are driving model behaviour. If tree models rely almost entirely on lag and rolling features, exogenous variables may be adding limited value.

## Recommended Next Phase

Do not add Prophet or scenario simulation until Phase 4 results have been reviewed. If feature models improve high-demand performance, the next phase should refine validation and feature selection. If they do not, the next step should investigate alternative target definitions, richer calendar features, weather data or a more careful statistical model search before adding new model families.
