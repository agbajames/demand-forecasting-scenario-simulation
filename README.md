# Demand Forecasting & Scenario Simulation – Python Time-Series Modelling

This repository is an **applied data science project** using real Great Britain electricity demand data to support operational demand forecasting and later scenario simulation.

## Business problem

Electricity system operators, suppliers, traders and planners need credible estimates of future demand to make operational decisions about generation scheduling, reserve margins, balancing actions and capacity risk. Demand varies by time of day, weekday, season, weather-linked behaviour, holidays, embedded generation and exceptional shocks. Poor demand forecasts can increase balancing costs, reduce resilience and make capacity planning less reliable.

## Why demand forecasting is an operational planning problem

Electricity must be balanced continuously. Unlike many products, large volumes of power cannot be stored cheaply at grid scale, so shortfalls or surpluses have immediate operational consequences. A practical forecasting workflow therefore needs to identify the correct target variable, understand the time frequency, quantify data quality issues and capture repeatable demand patterns before any forecasting model is trained.

## Data source

The primary source is the NESO Historic Demand Data package exposed through the CKAN API:

<https://api.neso.energy/api/3/action/datapackage_show?id=historic-demand-data>

NESO publishes Historic Demand Data as separate annual CSV resources rather than one single historic file. The ingestion script saves the package metadata, creates a resource inventory, identifies annual historic demand CSVs, downloads the selected year range and creates a combined raw CSV for EDA.

## Current phase

The first phase focused on project setup, NESO data ingestion, data profiling and deep exploratory data analysis. Phase 2 established clean baseline forecasting performance before any more advanced modelling was introduced. Phase 3 adds practical SARIMA and SARIMAX statistical forecasting, benchmarked against the seasonal naive baseline. Phase 3B diagnoses why the statistical models did not beat the benchmark and refines simple benchmark comparisons.

## Planned workflow

1. Ingest NESO package metadata and annual raw demand data.
2. Profile the raw dataset structure and data quality.
3. Discover actual date/time, demand and external-variable columns from the data.
4. Analyse missingness, duplicates, gaps, outliers and shocks.
5. Identify seasonal patterns and a suitable modelling frequency.
6. Produce written EDA conclusions and modelling recommendations.
7. Establish baseline forecasting performance.
8. Compare practical SARIMA and SARIMAX statistical models against the seasonal naive benchmark.
9. Diagnose model errors and refine benchmark understanding.
10. In later phases, compare more advanced forecasting models and add scenario simulation.

## Future modelling plan

Later phases may compare classical and modern time-series methods such as SARIMA/SARIMAX, Prophet, LSTM, Temporal Fusion Transformer, N-BEATS and Chronos. These models are intentionally not implemented in this phase. The first modelling phase should start with transparent statistical baselines and only add complex models if EDA shows that they are justified.

## Future simulation plan

A later Monte Carlo layer will simulate uncertainty around demand forecasts and estimate capacity-breach probability under selected capacity thresholds. Candidate assumptions may include residual error distributions, stress multipliers, seasonal volatility and demand shock scenarios, subject to findings from the EDA.

## Project structure

```text
data/
  raw/          # raw metadata and downloaded source files
  processed/    # future cleaned modelling-ready data
notebooks/
  01_deep_eda.ipynb
  02_baseline_forecasting.ipynb
  03_statistical_forecasting.ipynb
  04_model_diagnostics.ipynb
outputs/
  figures/eda/  # generated EDA charts
  figures/modelling/  # generated baseline forecasting charts
  tables/       # generated resource and EDA summary tables
reports/
  eda_summary.md
  baseline_forecasting_summary.md
  statistical_forecasting_summary.md
  model_diagnostics_summary.md
src/
  ingest_neso.py
  prepare_data.py
  baseline_models.py
  statistical_models.py
  model_diagnostics.py
  eda.py
  utils.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run ingestion

```bash
python src/ingest_neso.py
```

By default, ingestion downloads recent complete annual files from 2019 to the latest complete year available. With the current NESO resource list and a 2026 run date, this means 2019-2025; the partial 2026 resource is excluded unless explicitly requested.

Custom year ranges can be supplied from the command line:

```bash
python src/ingest_neso.py --start-year 2001 --end-year 2025
python src/ingest_neso.py --start-year 2024 --end-year 2026 --include-partial-current-year
```

Expected outputs include:

- `data/raw/neso_package_metadata.json`
- `outputs/tables/neso_resource_inventory.csv`
- `data/raw/selected_resource_info.json` documenting the selected annual resources
- annual raw downloaded NESO CSV files in `data/raw/`
- a combined raw dataset such as `data/raw/neso_historic_demand_2019_2025.csv`

## Run data preparation

After ingestion and EDA timestamp checks, create the duplicate-timestamp audit and processed daily modelling dataset:

```bash
python src/prepare_data.py
```

Optional arguments:

```bash
python src/prepare_data.py --target nd
python src/prepare_data.py --target tsd --output data/processed/daily_demand_2019_2025.csv
```

The preparation workflow preserves the raw annual CSVs and the combined raw half-hourly file. It uses `data/raw/selected_resource_info.json` to locate the combined raw dataset, validates or recreates `settlement_datetime`, writes `outputs/tables/duplicate_settlement_datetime_audit.csv`, resolves duplicated half-hourly timestamps with explicit aggregation rules, and saves `data/processed/daily_demand_2019_2025.csv`.

## Run baseline forecasting

Phase 2 establishes baseline forecasting performance on the processed daily dataset before SARIMAX, Prophet, machine learning or simulation models are considered.

```bash
python src/baseline_models.py --target nd_mean
```

The full current workflow is:

```bash
python src/ingest_neso.py
python src/prepare_data.py
python src/baseline_models.py --target nd_mean
```

Baseline outputs include:

- `outputs/tables/baseline_model_comparison.csv`
- `outputs/tables/baseline_forecasts.csv`
- `outputs/figures/modelling/actual_vs_baseline_forecasts.png`
- `outputs/figures/modelling/forecast_errors_by_model.png`

## Run statistical forecasting

Phase 3 compares SARIMA and SARIMAX against the seasonal naive benchmark from Phase 2. Do not describe SARIMA/SARIMAX as an improvement unless the saved metrics beat seasonal naive on the same 2025 test period.

```bash
python src/statistical_models.py --target nd_mean
```

The full modelling workflow is:

```bash
python src/ingest_neso.py
python src/prepare_data.py
python src/baseline_models.py --target nd_mean
python src/statistical_models.py --target nd_mean
```

Statistical forecasting outputs include:

- `outputs/tables/statistical_model_comparison.csv`
- `outputs/tables/statistical_forecasts.csv`
- `outputs/figures/modelling/actual_vs_statistical_forecasts.png`
- `outputs/figures/modelling/statistical_forecast_errors.png`
- `outputs/figures/modelling/model_mape_comparison.png`

## Run model diagnostics

Phase 3B diagnoses model errors and compares refined simple benchmarks. It does not add Prophet, simulation or deep learning.

```bash
python src/model_diagnostics.py --target nd_mean
```

The full workflow is:

```bash
python src/ingest_neso.py
python src/prepare_data.py
python src/baseline_models.py --target nd_mean
python src/statistical_models.py --target nd_mean
python src/model_diagnostics.py --target nd_mean
```

Diagnostic outputs include:

- `outputs/tables/error_summary_by_month.csv`
- `outputs/tables/error_summary_by_day_of_week.csv`
- `outputs/tables/error_summary_by_demand_regime.csv`
- `outputs/tables/incomplete_day_forecast_impact.csv`
- `outputs/tables/statistical_residual_autocorrelation.csv`
- `outputs/tables/exogenous_feature_correlation_with_targets.csv`
- `outputs/tables/refined_benchmark_comparison.csv`
- `outputs/figures/modelling/error_by_month.png`
- `outputs/figures/modelling/error_by_demand_regime.png`
- `outputs/figures/modelling/statistical_residual_autocorrelation.png`
- `outputs/figures/modelling/exogenous_target_correlation.png`
- `outputs/figures/modelling/refined_benchmark_comparison.png`

## Run the EDA notebook

Start Jupyter from the repository root:

```bash
jupyter notebook notebooks/01_deep_eda.ipynb
```

Then run the notebook cells in order after the ingestion script has completed. The notebook is designed to discover the actual NESO columns rather than assuming fixed names.

## Project limitations

- Annual NESO resource detection uses the published name and URL patterns and may require manual confirmation if the package structure changes.
- EDA recommendations are provisional until the selected annual raw files and combined dataset have been downloaded and profiled.
- Weather, holidays and market variables are not yet integrated.
- No forecasting model or simulation layer is included in this phase.
- Raw data files can be large and are not committed by default.
