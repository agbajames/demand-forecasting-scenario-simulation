# Scenario Simulation and Capacity-Pressure Summary

## Objective

Phase 5 adds a scenario simulation layer to the NESO demand forecasting project. The aim is to estimate demand ranges and capacity-pressure risk under simplified stress assumptions, using the validated forecasting outputs from the earlier modelling phases.

This phase supports the project title, Demand Forecasting & Scenario Simulation, while keeping a clear distinction between:

- operational one-day-ahead forecasting;
- strict recursive long-horizon forecasting;
- scenario simulation for decision-support stress testing.

## Base Forecast Used

The default base forecast is:

`gradient_boosting_forecast`

from:

`outputs/tables/feature_model_forecasts.csv`

This is the best-performing operational one-day-ahead feature model from the local Phase 4 results:

- MAE: 810.15
- RMSE: 1046.99
- MAPE: 3.09%

Random Forest was strongest on high-demand days in operational mode, with high-demand MAPE of 2.89%.

## Forecast-Design Caveat

The scenario layer inherits the forecast-design assumptions of the base forecast. The default Gradient Boosting forecast should be described as an operational one-day-ahead forecast baseline, because it can use recent actual demand history up to the previous day.

Strict recursive validation showed that year-over-year seasonal naive remained slightly stronger for long-horizon forecasting:

- seasonal naive MAPE: 7.93%
- strict recursive Gradient Boosting MAPE: 8.15%

Scenario simulation should therefore not be presented as a full-year-ahead forecast of actual grid stress. It is a stress-testing layer around a validated operational forecast baseline.

## Capacity Threshold Definition

By default, the capacity threshold is calculated as the 95th percentile of actual `nd_mean` in the training period before `2025-01-01`.

The threshold documentation is saved to:

`outputs/tables/capacity_threshold_documentation.csv`

If a real business or operational threshold is available, it can be supplied from the command line:

```bash
python src/scenario_simulation.py --target nd_mean --capacity-threshold 42000
```

## Scenario Definitions

The script creates five scenario definitions for the 2025 test period.

### Baseline

No systematic demand uplift is applied. Demand uncertainty comes from the empirical residual distribution estimated from operational Gradient Boosting forecast errors.

### High Demand Stress

Applies a 5% demand uplift on high-demand or peak-season days.

### Low Renewable Stress

Applies a 2% uplift on days with low embedded wind and solar generation, using the lower quartile of training-period embedded renewable generation as a proxy.

This is a simplified stress proxy. It is not a physical model of the electricity system, renewable dispatch, balancing actions or consumer behaviour.

### Winter Peak Stress

Applies a 7% uplift on winter or peak-season days.

### Combined Stress

Combines the high-demand, low-renewable and winter stress assumptions, with the total uplift capped at 12% to avoid unrealistic compounding.

## Simulation Method

For each scenario and each date in the 2025 test period, the simulation:

1. starts from the selected base forecast;
2. applies the scenario-specific uplift;
3. samples empirical residual noise from operational forecast errors;
4. creates repeated simulated demand values;
5. estimates the lower, median and upper demand range;
6. calculates the probability of exceeding the capacity threshold.

The default run uses 1,000 simulations per scenario-date and a fixed random seed of 42.

## Key Outputs

Daily scenario outputs are saved to:

`outputs/tables/scenario_daily_simulation_summary.csv`

Scenario-level capacity-pressure metrics are saved to:

`outputs/tables/scenario_capacity_pressure_summary.csv`

Figures are saved to:

- `outputs/figures/scenario_simulation/scenario_exceedance_probability.png`
- `outputs/figures/scenario_simulation/scenario_simulated_demand_ranges.png`
- `outputs/figures/scenario_simulation/scenario_capacity_pressure_summary.png`

## Interpreting Exceedance Probabilities

The exceedance probability is the share of simulated demand outcomes above the selected capacity threshold for a given date and scenario. For example, an exceedance probability of 0.30 means that 30% of simulated demand outcomes were above the threshold under that scenario's assumptions.

These probabilities are useful for comparing relative stress across scenarios. They should not be interpreted as precise probabilities of real-world system failure.

## Limitations

- The scenarios are simplified stress tests, not physical dispatch models.
- Exogenous effects are proxy assumptions, especially for low renewable generation.
- Results depend on the chosen forecast design and base forecast.
- The default base forecast is operational one-day-ahead, not a strict full-year-ahead model.
- Residual sampling assumes that historical forecast errors are informative for the simulated period.
- Weather, holidays, operational constraints and real capacity margins are not yet modelled directly.
- This phase does not add Prophet, deep learning or a production-grade probabilistic forecasting model.

## Recommended Next Phase

The next phase should focus on improving the realism of the scenario layer. Useful extensions include adding weather variables, holiday indicators, externally forecast renewable generation, and a clearer business-defined capacity threshold. Prophet or other advanced forecasting models can be considered later, but they should be benchmarked against the existing one-day-ahead and strict-recursive baselines rather than introduced as a shortcut.
