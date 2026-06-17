# Final Project Summary

## Executive Summary

This project builds an end-to-end electricity demand forecasting and scenario simulation workflow using NESO Historic Demand Data for Great Britain. It covers ingestion, deep EDA, data preparation, benchmark forecasting, SARIMA/SARIMAX statistical modelling, model diagnostics, feature-engineered operational forecasting, forecast-design validation and capacity-pressure scenario simulation.

The strongest operational model was a feature-engineered Gradient Boosting model with MAE 810.15, RMSE 1046.99 and MAPE 3.09% on the 2025 test period for `nd_mean`. Random Forest was strongest on high-demand days, with high-demand MAPE of 2.89%.

The project deliberately avoids overclaiming. The 3.09% result is an operational one-day-ahead forecast result, not a 365-day-ahead forecast from 1 January 2025. Strict recursive validation showed that seasonal naive remained slightly stronger overall than strict recursive Gradient Boosting.

## Data and Preparation

The source data is NESO Historic Demand Data, published as annual CSV resources. The ingestion workflow combines annual files for 2019-2025 into a raw half-hourly settlement-period dataset.

The local run produced:

- raw combined half-hourly dataset: 122,736 rows;
- processed daily modelling dataset: 2,557 rows and 32 columns;
- target used for main modelling: `nd_mean`;
- duplicate settlement timestamp audit: 28 rows across 14 timestamp groups;
- incomplete test day: 2025-10-26, with `coverage_ratio` 0.96.

The data preparation layer converts NESO half-hourly settlement data into a daily modelling dataset with mean and peak demand variables, renewable generation/capacity variables, interconnector variables, calendar features and data-quality flags.

## Forecasting Benchmark Discipline

The project starts with simple benchmarks before introducing more complex models. This is important because electricity demand has strong seasonal structure and simple seasonal baselines can be difficult to beat.

The seasonal naive model achieved MAE 2,076.58, RMSE 2,648.82 and MAPE 7.93% on the 2025 test period. This remained the key benchmark throughout the project.

## Statistical Modelling Findings

SARIMA and SARIMAX were tested after the baseline models. SARIMAX improved on SARIMA and naive forecasts, but it did not outperform seasonal naive.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| seasonal_naive | 2,076.58 | 2,648.82 | 7.93% |
| SARIMAX | 2,868.56 | 3,748.90 | 10.13% |
| SARIMA | 3,728.39 | 4,408.46 | 13.93% |
| naive | 3,797.90 | 4,641.25 | 14.24% |

The modelling conclusion is that statistical complexity alone was not enough. SARIMAX should be treated as an informative experiment rather than a performance improvement.

## Feature-Engineered Forecasting Findings

Feature-engineered models used lag features, rolling demand features, calendar features and renewable/exogenous variables. These models substantially improved operational one-day-ahead accuracy.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Gradient Boosting | 810.15 | 1,046.99 | 3.09% |
| Random Forest | 817.82 | 1,076.69 | 3.17% |
| Ridge | 965.24 | 1,208.91 | 3.79% |
| seasonal_naive benchmark | 2,076.58 | 2,648.82 | 7.93% |

High-demand regime results were particularly strong:

- Random Forest high-demand MAPE: 2.89%;
- Gradient Boosting high-demand MAPE: 3.41%;
- seasonal naive high-demand MAPE: 9.54%;
- SARIMAX high-demand MAPE: 19.71%.

## Forecast-Design Validation

Forecast-design validation was added to clarify what the feature-model results mean operationally.

The operational one-day-ahead setup is valid when recent actual demand history is available. This is realistic for next-day demand forecasting, because lag and rolling features can use data up to the previous day.

The strict recursive setup tests a harder long-horizon design. It forecasts through the test period sequentially and uses model forecasts, not actual future demand, to create future lag and rolling features.

| Forecast Design | Model | MAPE | Interpretation |
|---|---|---:|---|
| Operational one-day-ahead | Gradient Boosting | 3.09% | Strongest operational result. |
| Strict recursive | Gradient Boosting with observed exogenous variables dropped | 8.15% | Long-horizon recursive result. |
| Benchmark | seasonal naive | 7.93% | Slightly stronger than strict recursive Gradient Boosting overall. |

The final interpretation is clear: feature models are strongest for operational one-day-ahead forecasting, while seasonal naive remains slightly stronger for strict long-horizon recursive forecasting.

## Scenario Simulation Findings

The scenario simulation layer estimates demand ranges and capacity-pressure risk under simplified stress assumptions. The default base forecast is the operational one-day-ahead Gradient Boosting forecast.

The capacity threshold is 35,037.93, based on the 95th percentile of training-period actual `nd_mean` demand.

Key verified scenario findings:

- baseline expected exceedance days: 7.07;
- combined stress expected exceedance days: 54.04;
- combined stress simulated peak p50: 41,028.83;
- combined stress simulated peak p95: 42,738.07;
- highest-risk periods cluster in winter high-demand days, especially January and February 2025.

The scenario simulation should be described as decision-support stress testing. It is not a physical grid-dispatch model and does not predict real-world system stress with certainty.

## Limitations

- Scenario assumptions are simplified stress-test assumptions.
- Observed exogenous variables must be known, forecasted or scenario-specified for deployment.
- Weather and holiday variables are not yet integrated.
- Strict recursive feature models do not beat seasonal naive overall.
- The capacity threshold is a percentile-based proxy, not an operational capacity limit.
- The project is not production-ready.

## Recommended Next Steps

1. Add weather and holiday features.
2. Test Prophet only after preserving the current benchmark discipline.
3. Tune feature models and evaluate calibration.
4. Replace observed exogenous variables with forecasted or scenario-specified inputs.
5. Use a business-defined capacity threshold if available.
6. Expand probabilistic evaluation for scenario outputs.

## Portfolio/CV Bullet Suggestions

These can be adapted for a CV, portfolio case study or interview discussion.

## CV-ready bullets

- Built an end-to-end electricity demand forecasting pipeline using NESO half-hourly demand data, combining 2019-2025 annual CSV resources into a validated daily modelling dataset.
- Benchmarked Naive, Seasonal Naive, Holt-Winters, SARIMA and SARIMAX models, retaining the seasonal naive benchmark where statistical models failed to improve performance.
- Developed feature-engineered one-day-ahead Gradient Boosting and Random Forest models using lag, rolling, calendar and renewable-generation features, reducing operational MAPE from 7.93% to 3.09%.
- Added forecast-design validation and Monte Carlo scenario simulation to estimate capacity-pressure risk, showing expected threshold exceedance days rising from around 7 under baseline assumptions to around 54 under combined stress.
