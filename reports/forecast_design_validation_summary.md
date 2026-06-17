# Forecast Design Validation Summary

## Executive Summary

Phase 4 feature-engineered models produced a strong improvement over the seasonal naive and SARIMAX benchmarks on the 2025 test period. The best overall model was gradient boosting, with MAE 810.15, RMSE 1046.99 and MAPE 3.09% on `nd_mean`. However, these results need to be described precisely: they represent an operational one-day-ahead forecast design, not a strict full-year-ahead forecast made on 1 January 2025.

Phase 4B adds a feature availability audit and strict recursive evaluation so the project can distinguish between valid next-day operational forecasting and longer-horizon recursive forecasting.

## Why This Validation Was Needed

The Phase 4 models use lag and rolling target features such as `lag_1`, `lag_7`, `rolling_7_mean` and `rolling_7_max`. These features are appropriate when forecasting tomorrow because yesterday and earlier demand values are already known. They are not automatically valid for a strict multi-step forecast across the whole 2025 test period unless future lag values are generated from earlier model forecasts.

This distinction matters for portfolio presentation. The strong Phase 4 results should not be hidden, but they should be labelled as one-day-ahead operational results.

## Current Interpretation of Phase 4 Feature Results

The current interpretation is:

- Feature-engineered one-day-ahead models achieved substantially lower error than seasonal naive on the 2025 test period.
- Gradient boosting was the strongest overall model in the local Phase 4 run, with MAPE 3.09%.
- Random forest was the strongest high-demand-regime model, with MAPE 2.89%.
- The result is operationally meaningful for next-day forecasting because recent actual demand history is realistically available.
- The result should not be described as a 365-day-ahead forecast.

Recommended wording:

> Feature-engineered one-day-ahead models achieved substantially lower error than seasonal naive on the 2025 test period. A separate strict recursive evaluation was added to distinguish operational next-day forecasting from longer-horizon forecasting.

## Feature Leakage and Availability Findings

The audit created by `src/forecast_validation.py` classifies each feature used by `src/feature_models.py`:

- Target lag features are safe for one-day-ahead forecasting if actual demand is known up to the previous day.
- Rolling target features are safe for one-day-ahead forecasting because they are shifted and exclude the forecast day.
- Target lag and rolling features must be recursively generated for strict multi-step forecasting.
- Calendar features are safe across forecast horizons.
- Same-day observed generation and interconnector variables are only safe if they are known, forecasted or scenario-specified at forecast time.
- Embedded wind and solar capacity variables may be easier to know in advance than realised generation, but should still be validated before operational deployment.

The audit is saved to:

`outputs/tables/feature_availability_audit.csv`

## Operational One-Day-Ahead Performance

Local Phase 4 results on `nd_mean` were:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| gradient_boosting | 810.15 | 1046.99 | 3.09% |
| random_forest | 817.82 | 1076.69 | 3.17% |
| ridge | 965.24 | 1208.91 | 3.79% |
| seasonal_naive | 2076.58 | 2648.82 | 7.93% |
| SARIMAX | 2868.56 | 3748.90 | 10.13% |

These metrics should be presented as operational one-day-ahead results because actual demand history up to the previous day is available for each forecast date.

## Strict Recursive Performance

The strict recursive evaluation trains before the test period and forecasts through 2025 sequentially. After each forecasted test day, lag and rolling target features use the model forecast rather than the actual target value.

The default strict recursive mode is:

```bash
python src/forecast_validation.py --target nd_mean --strict-exog-mode drop
```

This drops same-day observed exogenous generation and flow variables from the strict recursive run unless they are forecasted or scenario-specified. A retrospective oracle mode is also available:

```bash
python src/forecast_validation.py --target nd_mean --strict-exog-mode actual
```

Strict recursive outputs are saved to:

- `outputs/tables/strict_recursive_feature_forecasts.csv`
- `outputs/tables/strict_recursive_feature_comparison.csv`
- `outputs/tables/forecast_design_comparison.csv`
- `outputs/tables/strict_recursive_regime_comparison.csv`

The strict recursive metrics should be filled in after running the script locally with the processed daily dataset.

## Benchmark Comparison

The key validation question is whether the feature models still beat the seasonal naive benchmark under stricter assumptions. The script saves this comparison in:

`outputs/tables/forecast_design_comparison.csv`

If strict recursive performance is weaker, this should be reported directly. Weaker strict-recursive results would not invalidate the one-day-ahead result; they would clarify that a different deployment design is needed for longer forecast horizons.

## Portfolio and README Wording Recommendation

Use wording that distinguishes forecast horizons clearly:

> Feature-engineered one-day-ahead models achieved substantially lower error than seasonal naive on the 2025 test period. A separate strict recursive evaluation was added to distinguish operational next-day forecasting from longer-horizon forecasting.

Avoid wording such as "the model forecasts all of 2025 from January 1st" unless strict recursive results are being reported.

## Risks and Limitations

- Same-day observed exogenous variables can create retrospective leakage unless replaced by forecasts, schedules or scenario assumptions.
- Strict recursive performance may degrade because forecast errors accumulate through lag and rolling features.
- Calendar variables remain safe, but weather and holidays are not yet included.
- This phase does not add Prophet, deep learning or Monte Carlo simulation.
- The strict recursive evaluation should be rerun whenever feature engineering changes.

## Next Steps

1. Run `python src/forecast_validation.py --target nd_mean`.
2. Compare operational and strict recursive metrics.
3. Update this report with the generated strict recursive results.
4. Decide whether the next modelling phase should focus on weather/holiday regressors, forecasted exogenous inputs, or Prophet/SARIMAX refinements.
