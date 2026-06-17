# Project Card

## Demand Forecasting & Scenario Simulation - Python

## Objective

Build an end-to-end applied data science workflow for Great Britain electricity demand forecasting and scenario-based capacity-pressure analysis.

## Dataset

NESO Historic Demand Data, using annual CSV resources for 2019-2025. The raw half-hourly settlement-period data is converted into a daily modelling dataset with demand, calendar, renewable generation/capacity, interconnector and data-quality features.

## Methods

- NESO CKAN API ingestion and annual CSV combination.
- Deep EDA and data-quality profiling.
- Daily data preparation from half-hourly settlement data.
- Naive, seasonal naive and Holt-Winters baselines.
- SARIMA and SARIMAX statistical forecasting.
- Feature-engineered Ridge, Random Forest and Gradient Boosting models.
- Forecast-design validation for operational one-day-ahead versus strict recursive forecasting.
- Monte Carlo scenario simulation for capacity-pressure analysis.

## Key Results

- Seasonal naive benchmark: 7.93% MAPE on the 2025 test period.
- Operational one-day-ahead Gradient Boosting: 3.09% MAPE.
- Operational Random Forest high-demand MAPE: 2.89%.
- SARIMAX did not beat seasonal naive overall, with 10.13% MAPE.

## Forecast-Design Caveat

The 3.09% Gradient Boosting result is an operational one-day-ahead result. It uses recent actual demand history, which is valid for next-day forecasting. It is not a full-year-ahead forecast from 1 January 2025. In strict recursive mode, seasonal naive remains slightly stronger overall than Gradient Boosting.

## Scenario Simulation Result

Using the 95th percentile of training-period `nd_mean` as a capacity-pressure threshold, expected exceedance days rose from around 7 under baseline assumptions to around 54 under combined stress. Highest-risk dates clustered around winter high-demand periods.

## Technologies Used

Python, pandas, NumPy, requests, matplotlib, seaborn, statsmodels, scipy, scikit-learn, Jupyter.

## Limitations

- Scenario assumptions are simplified stress tests.
- This is not a physical grid-dispatch model.
- Observed exogenous variables must be known, forecasted or scenario-specified for deployment.
- Weather and holiday variables are not yet integrated.
- The project is not production-ready.

## Next Steps

Add weather and holiday features, improve exogenous-variable availability assumptions, tune feature models, test Prophet carefully against the existing benchmarks and refine probabilistic scenario calibration.
