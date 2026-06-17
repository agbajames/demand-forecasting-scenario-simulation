# Results Index

This index explains the main generated outputs from the demand forecasting and scenario simulation workflow. Most files are created locally and ignored by Git.

## Tables

| Output | Meaning |
|---|---|
| `outputs/tables/baseline_model_comparison.csv` | Compares naive, seasonal naive and Holt-Winters baseline forecast accuracy. |
| `outputs/tables/statistical_model_comparison.csv` | Compares SARIMA/SARIMAX results against baseline metrics. |
| `outputs/tables/error_summary_by_demand_regime.csv` | Summarises forecast errors by low, normal and high demand regimes. Useful for checking peak-demand weakness. |
| `outputs/tables/feature_model_comparison.csv` | Compares Ridge, Random Forest and Gradient Boosting feature-model accuracy against earlier benchmarks. |
| `outputs/tables/feature_model_regime_comparison.csv` | Compares feature-model accuracy by demand regime, including high-demand days. |
| `outputs/tables/forecast_design_comparison.csv` | Compares operational one-day-ahead and strict recursive forecast designs, including the seasonal naive benchmark. |
| `outputs/tables/scenario_capacity_pressure_summary.csv` | Scenario-level capacity-pressure metrics, including expected exceedance days and peak simulated demand ranges. |
| `outputs/tables/scenario_daily_simulation_summary.csv` | Daily scenario simulation output with simulated demand intervals and capacity-threshold exceedance probabilities. |

## Figure Directories

| Output | Meaning |
|---|---|
| `outputs/figures/modelling/` | Forecasting and diagnostic plots for baseline, statistical, feature-model and forecast-design validation phases. |
| `outputs/figures/scenario_simulation/` | Scenario simulation plots showing exceedance probabilities, simulated demand ranges and scenario-level capacity pressure. |

## How to Regenerate Outputs

Run the full workflow from the repository root:

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

Generated raw data, processed data, tables and figures are ignored by Git where appropriate.
