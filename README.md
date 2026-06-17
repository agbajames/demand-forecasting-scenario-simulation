# Demand Forecasting & Scenario Simulation - Python

## Executive Summary

This applied data science project builds a complete electricity demand forecasting and scenario simulation workflow using NESO Historic Demand Data for Great Britain. It moves from ingestion and exploratory data analysis through data preparation, benchmark forecasting, SARIMA/SARIMAX statistical modelling, feature-engineered operational forecasting, forecast-design validation and capacity-pressure scenario simulation.

The strongest operational result is a feature-engineered Gradient Boosting model with 3.09% MAPE on the 2025 test period. The project also explicitly validates the forecast design: this result is an operational one-day-ahead forecast, not a 365-day-ahead forecast from 1 January 2025. For strict long-horizon recursive forecasting, the year-over-year seasonal naive benchmark remains slightly stronger overall.

The scenario layer uses the validated operational forecast as a stress-testing baseline. Using the 95th percentile of training-period demand as a capacity-pressure threshold, the simulation estimated around 7 expected exceedance days under the baseline scenario, rising to around 54 expected exceedance days under a combined stress scenario. Highest-risk dates clustered around winter high-demand periods.

## Problem Framing

Electricity demand forecasting matters because power systems must balance supply and demand continuously. Better forecasts support generation scheduling, reserve planning, balancing decisions, capacity-risk monitoring and operational resilience.

High-demand and capacity-pressure days matter because errors during peak conditions are more consequential than errors on normal days. A model that performs well on average can still be operationally weak if it misses winter peaks or high-demand regimes.

Benchmark discipline is central to the project. Naive and seasonal naive models are simple, but they provide a hard-to-beat reference point. This project keeps the seasonal naive benchmark when SARIMA/SARIMAX models fail to improve performance, rather than overstating model complexity.

Scenario simulation adds decision-support value by translating forecast uncertainty and stress assumptions into capacity-threshold exceedance probabilities. It does not replace forecast evaluation and it is not a physical grid-dispatch model.

## Data Source

The project uses NESO Historic Demand Data:

<https://api.neso.energy/api/3/action/datapackage_show?id=historic-demand-data>

NESO publishes the data as annual CSV resources. The ingestion layer combines annual files for 2019-2025 into a raw half-hourly settlement-period dataset. The local EDA run confirmed:

- raw combined half-hourly dataset: 122,736 rows;
- processed daily modelling dataset: 2,557 rows and 32 columns;
- main modelling target: `nd_mean`;
- duplicate settlement timestamp audit: 28 rows across 14 timestamp groups;
- incomplete test day: 2025-10-26, with `coverage_ratio` 0.96.

Raw and processed data outputs are generated locally and are not committed to Git.

## Project Workflow

Run the workflow from the repository root:

```bash
python src/ingest_neso.py
python src/prepare_data.py
python src/baseline_models.py --target nd_mean
python src/statistical_models.py --target nd_mean
python src/model_diagnostics.py --target nd_mean
python src/feature_models.py --target nd_mean
python src/forecast_validation.py --target nd_mean
python src/scenario_simulation.py --target nd_mean --n-simulations 1000
```

## Key Results

### Baseline and Statistical Models

2025 test-period results on `nd_mean`:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| seasonal_naive | 2,076.58 | 2,648.82 | 7.93% |
| SARIMAX | 2,868.56 | 3,748.90 | 10.13% |
| SARIMA | 3,728.39 | 4,408.46 | 13.93% |
| naive | 3,797.90 | 4,641.25 | 14.24% |

SARIMAX improved over SARIMA and naive forecasts, but it did not beat the seasonal naive benchmark.

### Feature-Engineered Operational One-Day-Ahead Models

2025 test-period operational results on `nd_mean`:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Gradient Boosting | 810.15 | 1,046.99 | 3.09% |
| Random Forest | 817.82 | 1,076.69 | 3.17% |
| Ridge | 965.24 | 1,208.91 | 3.79% |
| seasonal_naive benchmark | 2,076.58 | 2,648.82 | 7.93% |

High-demand regime results:

| Model | High-Demand MAPE |
|---|---:|
| Random Forest | 2.89% |
| Gradient Boosting | 3.41% |
| seasonal_naive | 9.54% |
| SARIMAX | 19.71% |

### Forecast-Design Validation

| Forecast Design | Model | MAPE | Interpretation |
|---|---|---:|---|
| Operational one-day-ahead | Gradient Boosting | 3.09% | Uses actual demand history up to the previous day. Valid for next-day operations. |
| Strict recursive | Gradient Boosting, observed exogenous variables dropped | 8.15% | Future lag features are generated recursively from model forecasts. |
| Benchmark | seasonal_naive | 7.93% | Slightly stronger than strict recursive Gradient Boosting overall. |

### Scenario Simulation

Capacity threshold: 35,037.93, defined as the 95th percentile of training-period actual `nd_mean` demand.

| Scenario | Stress Assumption | Expected Exceedance Days | Notes |
|---|---|---:|---|
| baseline | No systematic uplift; residual uncertainty only | 7.07 | Reference scenario. |
| low_renewable_stress | Simplified +2% proxy uplift on low embedded wind/solar days | See `outputs/tables/scenario_capacity_pressure_summary.csv` | Proxy assumption, not a physical renewable model. |
| high_demand_stress | +5% uplift on high-demand or peak-season days | See `outputs/tables/scenario_capacity_pressure_summary.csv` | Tests peak-demand sensitivity. |
| winter_peak_stress | +7% uplift on winter or peak-season days | See `outputs/tables/scenario_capacity_pressure_summary.csv` | Focuses on winter capacity pressure. |
| combined_stress | Combined stress assumptions with a 12% uplift cap | 54.04 | Simulated peak p50: 41,028.83; simulated peak p95: 42,738.07. |

Highest-risk periods clustered around winter high-demand days, especially January and February 2025.

## Forecast-Design Caveat

The 3.09% MAPE Gradient Boosting result is an operational one-day-ahead result. It uses recent actual demand history, which is valid for next-day operational forecasting because yesterday and earlier demand values are known at forecast time.

It is not a 365-day-ahead forecast made from 1 January 2025. Strict recursive validation was added to avoid leakage and overclaiming. In strict recursive mode, future lag and rolling features are generated from model forecasts rather than actual future target values. Under that stricter long-horizon design, seasonal naive remains slightly stronger overall than Gradient Boosting.

## Scenario Simulation Interpretation

Using the 95th percentile of training-period demand as a capacity-pressure threshold, the scenario simulation estimated around 7 expected exceedance days under the baseline scenario, rising to around 54 expected exceedance days under a combined stress scenario. Highest-risk dates clustered around winter high-demand periods.

The simulation is a decision-support stress-testing layer. It estimates demand ranges and threshold exceedance probabilities under simplified assumptions; it does not predict actual future grid stress with certainty.

## Limitations

- Scenario assumptions are simplified stress-test assumptions.
- This is not a physical grid-dispatch model.
- Observed exogenous variables must be known, forecasted or scenario-specified for future deployment.
- Strict recursive feature models do not beat seasonal naive overall in the current validation.
- Weather variables, holiday effects and real operational capacity margins are not yet integrated.
- Further work could add Prophet, weather variables, holiday features, hyperparameter tuning and probabilistic calibration.
- The project is not production-ready; it is an applied analytics workflow and portfolio project.

## Repository Structure

```text
data/
  raw/                 # generated raw metadata and NESO source files, ignored where appropriate
  processed/           # generated modelling-ready datasets, ignored where appropriate
notebooks/
  01_deep_eda.ipynb
  02_baseline_forecasting.ipynb
  03_statistical_forecasting.ipynb
  04_model_diagnostics.ipynb
  05_feature_modelling.ipynb
  06_forecast_design_validation.ipynb
  07_scenario_simulation.ipynb
outputs/
  figures/eda/
  figures/modelling/
  figures/scenario_simulation/
  tables/
reports/
  eda_summary.md
  baseline_forecasting_summary.md
  statistical_forecasting_summary.md
  model_diagnostics_summary.md
  feature_modelling_summary.md
  forecast_design_validation_summary.md
  scenario_simulation_summary.md
  final_project_summary.md
  results_index.md
  project_card.md
src/
  ingest_neso.py
  prepare_data.py
  baseline_models.py
  statistical_models.py
  model_diagnostics.py
  feature_models.py
  forecast_validation.py
  scenario_simulation.py
  eda.py
  utils.py
```

## Reproducibility

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Then run:

```bash
python src/ingest_neso.py
python src/prepare_data.py
python src/baseline_models.py --target nd_mean
python src/statistical_models.py --target nd_mean
python src/model_diagnostics.py --target nd_mean
python src/feature_models.py --target nd_mean
python src/forecast_validation.py --target nd_mean
python src/scenario_simulation.py --target nd_mean --n-simulations 1000
```

Generated raw data, processed data, tables and figures are produced locally and ignored by Git where appropriate.
