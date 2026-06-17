# Demand Forecasting & Scenario Simulation – Python Time-Series Modelling

This repository is an **applied data science project** using real Great Britain electricity demand data to support operational demand forecasting and later scenario simulation.

## Business problem

Electricity system operators, suppliers, traders and planners need credible estimates of future demand to make operational decisions about generation scheduling, reserve margins, balancing actions and capacity risk. Demand varies by time of day, weekday, season, weather-linked behaviour, holidays, embedded generation and exceptional shocks. Poor demand forecasts can increase balancing costs, reduce resilience and make capacity planning less reliable.

## Why demand forecasting is an operational planning problem

Electricity must be balanced continuously. Unlike many products, large volumes of power cannot be stored cheaply at grid scale, so shortfalls or surpluses have immediate operational consequences. A practical forecasting workflow therefore needs to identify the correct target variable, understand the time frequency, quantify data quality issues and capture repeatable demand patterns before any forecasting model is trained.

## Data source

The primary source is the NESO Historic Demand Data package exposed through the CKAN API:

<https://api.neso.energy/api/3/action/datapackage_show?id=historic-demand-data>

The ingestion script saves the package metadata, creates a resource inventory and attempts to download the most relevant raw CSV demand-observation resource without overwriting or transforming the raw source content.

## Current phase

This first phase focuses only on project setup, NESO data ingestion, data profiling and deep exploratory data analysis. It deliberately avoids forecasting models and scenario simulation code until the dataset is understood.

## Planned workflow

1. Ingest NESO package metadata and raw demand data.
2. Profile the raw dataset structure and data quality.
3. Discover actual date/time, demand and external-variable columns from the data.
4. Analyse missingness, duplicates, gaps, outliers and shocks.
5. Identify seasonal patterns and a suitable modelling frequency.
6. Produce written EDA conclusions and modelling recommendations.
7. In later phases, compare forecasting models and add scenario simulation.

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
outputs/
  figures/eda/  # generated EDA charts
  tables/       # generated resource and EDA summary tables
reports/
  eda_summary.md
src/
  ingest_neso.py
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

Expected outputs include:

- `data/raw/neso_package_metadata.json`
- `outputs/tables/neso_resource_inventory.csv`
- `data/raw/selected_resource_info.json` if a resource is selected confidently
- a raw downloaded NESO data file in `data/raw/` if a suitable resource is identified

## Run the EDA notebook

Start Jupyter from the repository root:

```bash
jupyter notebook notebooks/01_deep_eda.ipynb
```

Then run the notebook cells in order after the ingestion script has completed. The notebook is designed to discover the actual NESO columns rather than assuming fixed names.

## Project limitations

- The raw NESO resource selection is heuristic and may require manual confirmation if the package structure changes.
- EDA recommendations are provisional until the latest raw data file has been downloaded and profiled.
- Weather, holidays and market variables are not yet integrated.
- No forecasting model or simulation layer is included in this phase.
- Raw data files can be large and are not committed by default.
