# EDA Summary – NESO Historic Demand Data

## Executive summary

This EDA phase is designed to establish a reliable foundation before any forecasting or simulation work begins. It focuses on NESO historic demand metadata ingestion, raw data discovery, column profiling, target-variable selection and data-quality assessment. The recommended target variable, modelling frequency and candidate external regressors should be confirmed by running `notebooks/01_deep_eda.ipynb` against the downloaded raw NESO file.

This EDA phase suggests that daily demand may be a practical modelling frequency for the first version of the forecasting pipeline if the downloaded data is sub-daily and noisy. It can preserve the main operational demand pattern while reducing computational complexity. However, half-hourly or hourly modelling should remain under consideration if data quality, business requirements and seasonality diagnostics support it. The recommended target variable is subject to confirmation from the discovered NESO columns, missingness checks and data dictionary review.

## Dataset overview

The dataset is sourced from the NESO Historic Demand Data CKAN package. The ingestion script stores full package metadata, creates a clean resource inventory and attempts to download a relevant raw CSV file containing historic demand observations. The EDA notebook then inspects the actual raw file structure rather than assuming specific column names.

## Data quality findings

The notebook should be used to document:

- column names, data types and example values;
- missing values by column and over time;
- duplicate rows and duplicate timestamps;
- timestamp gaps relative to the inferred frequency;
- suspicious zero or negative demand values;
- unusually high or low demand observations.

Rows and columns should not be dropped silently. Any exclusions in later phases should be explicit, reproducible and justified.

## Recommended target variable

The primary target should be selected from numeric columns whose names and profiles indicate demand concepts such as national demand, transmission system demand, total demand or electricity load. Selection should consider operational meaning, continuity, missingness, outlier behaviour and whether the series aligns with the intended forecasting question.

## Recommended modelling frequency

The recommended modelling frequency should be based on the inferred timestamp interval, missingness, duplicate timestamps, operational use case and seasonal diagnostics. Candidate frequencies include half-hourly, hourly, daily and weekly. Daily modelling may be an appropriate first baseline if it reduces sub-daily noise while retaining useful seasonality.

## Key seasonal patterns

The notebook analyses daily, day-of-week, weekday-versus-weekend, monthly and year-on-year patterns where supported by the timestamp frequency. Later modelling should encode the strongest repeatable patterns identified during this phase.

## Outliers and anomalies

Extreme observations should be detected using IQR and, where useful, z-score logic. The EDA should flag unusually high or low demand periods and visible shock periods such as COVID-era changes if the date range includes them. Outliers should be documented, not automatically removed.

## Candidate external variables

Potential external regressors already present in the NESO data may include embedded wind generation, embedded solar generation, interconnector flows, hydro storage pumping, STOR and other supply or balancing variables. Candidate variables should be assessed using missingness, correlation with the target and simple lag checks. The strongest candidates can be considered later for SARIMAX-style models.

## Recommended train/test and rolling-origin validation setup

A chronological split should be used, avoiding random shuffling. A sensible first approach is to reserve the most recent contiguous period as a holdout set and use rolling-origin validation on the earlier history. The exact split should reflect the selected modelling frequency and the amount of clean history available.

## Recommended scenario simulation assumptions

Initial simulation assumptions should be grounded in EDA findings, including target volatility, seasonal variance, outlier frequency and shock periods. Later Monte Carlo scenarios may use residual distributions from fitted models, stress multipliers, seasonal demand uplift factors and explicit low/high demand cases.

## Capacity breach probability approach

A later capacity-risk layer can compare simulated demand paths with one or more capacity thresholds. The output should be the estimated probability that demand exceeds a chosen threshold over a defined horizon, with sensitivity analysis across threshold levels and stress assumptions.

## Risks and limitations

- Resource selection depends on NESO package metadata and may need manual review.
- Column names and meanings must be verified from the actual data and NESO documentation.
- Demand can be affected by weather, holidays, economic activity and exceptional shocks not yet integrated.
- Correlation analysis does not prove causal relationships.
- Deep learning models may be unnecessary unless simpler baselines underperform and sufficient clean history is available.

## Next steps

1. Run `python src/ingest_neso.py`.
2. Open and execute `notebooks/01_deep_eda.ipynb` from top to bottom.
3. Confirm the recommended demand target and modelling frequency.
4. Update this report with concrete EDA outputs from the downloaded dataset.
5. Build baseline forecasting models only after the EDA findings are reviewed.
