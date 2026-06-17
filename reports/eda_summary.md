# EDA Summary - NESO Historic Demand Data

## Executive Summary

The EDA was run against the combined NESO Historic Demand dataset covering 2019-2025. The dataset contains 122,736 rows and 23 columns, representing half-hourly settlement-period observations rather than one observation per calendar date.

The main data preparation finding is that `settlement_date` alone is not a valid modelling timestamp. Repeated `settlement_date` values are expected because each day contains multiple settlement periods. The correct timestamp for time-series analysis is `settlement_datetime`, created from `settlement_date` and `settlement_period`.

This phase remains limited to data ingestion, profiling and EDA. No forecasting model, Prophet, SARIMAX or simulation layer has been added.

## Dataset Overview

The source data is the NESO Historic Demand Data package. The ingestion workflow now combines annual CSV resources into `data/raw/neso_historic_demand_2019_2025.csv`, while retaining the raw annual files separately.

The current combined dataset covers selected years 2019-2025 and contains half-hourly settlement data. Settlement period 1 maps to 00:00, period 2 to 00:30 and period 48 to 23:30. Clock-change days may contain non-standard settlement-period counts, so these cases should be inspected rather than removed automatically.

## Data Quality Findings

The local EDA found no missing values in the main candidate demand target columns: `nd`, `tsd` and `england_wales_demand`.

Duplicate `settlement_date` values are expected and should not be treated as duplicate observations. Duplicate timestamp checks should instead use `settlement_datetime`, which combines settlement date and period into a half-hourly time index.

No zero or negative values were found in the provisional target `nd`. IQR screening detected 226 high-demand observations. These should be documented and monitored in later modelling rather than removed automatically.

COVID-era rows are present in the 2019-2025 dataset. They should be retained because they are genuine historical observations, but later validation and scenario design should consider whether COVID-era demand behaviour creates structural-break risk.

## Recommended Target Variable

The candidate demand targets are:

- `nd`
- `tsd`
- `england_wales_demand`

The current recommendation is to choose either `nd` or `tsd`, depending on the operational question being modelled. `nd` is a strong candidate for a national demand forecasting target and has no missing or non-positive values in the current EDA. `tsd` may be preferable if the intended use case is closer to transmission-system demand and operational balancing requirements.

`england_wales_demand` is also a legitimate demand series, but it is geographically narrower and may be less suitable if the project objective is GB-wide operational demand planning.

Embedded wind generation, embedded wind capacity, embedded solar generation and embedded solar capacity should not be treated as demand targets. They are candidate external variables.

## Recommended Modelling Frequency

The native NESO data is half-hourly. `settlement_datetime` should be used as the modelling timestamp for all native-frequency analysis.

Daily aggregation may still be a practical first modelling frequency. It can reduce half-hourly volatility and computational complexity while preserving major weekly, seasonal and annual demand patterns. The raw half-hourly data should remain available so later phases can compare daily baselines with higher-resolution modelling if needed.

## Key Seasonal Patterns

The dataset supports analysis of intraday, day-of-week, monthly and year-on-year demand patterns. Half-hourly observations are especially useful for identifying within-day peaks and troughs, while daily aggregation may be useful for a first modelling baseline.

Later modelling should encode the strongest repeatable patterns identified in the notebook, including daily shape, weekday/weekend differences and seasonal demand variation.

## Outliers and Anomalies

IQR screening found 226 high-demand observations. These should be reviewed in context, particularly around winter peak demand periods, but should not be removed automatically.

COVID-era observations are present and should be retained. They may affect model validation and should be monitored as a potential shock or structural-break period.

## Candidate External Variables

Likely external variables already available in the NESO dataset include:

- embedded wind generation
- embedded wind capacity
- embedded solar generation
- embedded solar capacity
- pumped storage or hydro pumping variables
- interconnector flow variables
- other supply or balancing fields present in the annual files

These variables may be useful later for SARIMAX-style regressors or scenario analysis, subject to missingness, correlation and lag diagnostics.

## Recommended Train/Test and Rolling-Origin Validation Setup

Use chronological splits only. Random shuffling would leak future information into training data.

A sensible first approach is to reserve the most recent contiguous period as a holdout set, then use rolling-origin validation over earlier history. If the first modelling version uses daily aggregation, the validation folds should also be daily. If later phases retain half-hourly resolution, the validation design should respect the native half-hourly timestamp sequence.

## Recommended Scenario Simulation Assumptions

Scenario assumptions should be grounded in the EDA findings: seasonal demand variation, half-hourly volatility, winter peak behaviour, COVID-era shock periods and the observed frequency of high-demand outliers.

The raw half-hourly dataset should be retained so later scenario work can test whether capacity-risk assumptions differ between daily and half-hourly demand views.

## Capacity Breach Probability Approach

A later capacity-risk layer can compare simulated demand paths with one or more capacity thresholds. The output should estimate the probability that demand exceeds a threshold over a defined horizon, with sensitivity analysis across threshold levels, seasonal stress assumptions and demand-shock scenarios.

No Monte Carlo simulation has been implemented in this phase.

## Risks and Limitations

- `settlement_datetime` must be used for time-series checks; `settlement_date` alone is not unique.
- Clock-change days may contain non-standard settlement-period counts and should be reviewed carefully.
- Target choice between `nd` and `tsd` should be confirmed against the NESO data dictionary and the intended operational use case.
- COVID-era behaviour may affect validation results and model stability.
- Correlation analysis for external variables does not prove causality.
- Weather, holidays and wider economic drivers are not yet integrated.

## Next Steps

1. Re-run `python src/ingest_neso.py` if the raw annual files or combined dataset need refreshing.
2. Re-run `notebooks/01_deep_eda.ipynb` from top to bottom so the notebook uses `settlement_datetime`.
3. Confirm the target choice between `nd` and `tsd`.
4. Keep outliers and COVID-era rows in the dataset, but document their effect on model validation.
5. Build baseline forecasting models only after the corrected EDA outputs are reviewed.
