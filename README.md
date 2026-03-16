# interfere_experiments

An experiment runner and benchmark generator for the [interfere](https://github.com/djpasseyjr/interfere) package. It provides reusable data generators, a standardized control-vs-response forecasting format, and the pipeline used to build the **Interfere Benchmark** systematically.

## What this package provides

- `**interfere_experiments.data_generators`** — A large library of `**DataGenerator**` subclasses. Each one wraps an interfere dynamics model with fixed parameters, observational and interventional setups, initial conditions, and timestep. You can generate train/forecast/intervention-response data with a single call (including optional burn-in and downsampling for stiff dynamics).
- `**interfere_experiments.control_vs_resp**` — The **control-vs-response** problem: `ControlVsRespData` (train states, forecast states, intervention response states, plus prior and intervention metadata) and helpers to **generate** data from any interfere model and to **export/import** it as JSON with full metadata.
- `**interfere_experiments.quick_models`** — Small helpers that instantiate specific interfere models used in notebooks (e.g. gut-check coupled logistic map and Belozyorov).

The package depends on **interfere** for dynamics and interventions; it is intended for running and reproducing experiments (e.g. forecasting and causal prediction comparisons) and for generating the Interfere Benchmark datasets.

## How the Interfere Benchmark was constructed

The **Interfere Benchmark** (version 1.1.1) is built in the notebook `notebooks/GeneratingInterfereBenchmark1.1.1.ipynb` using the following techniques:

1. **Curated model set** — A fixed list of `**DataGenerator`** classes (`BENCHMARK_MODELS`) is defined from this package. It spans low- and high-dimensional, discrete- and continuous-time, and deterministic and stochastic dynamics (e.g. AttractingFixedPoint4D, Belozyorov1–3, CoupledLogisticMap variants, CoupledMapLattice variants, Kuramoto/Kuramoto–Sakaguchi, Lorenz, Rössler, Lotka–Volterra, SIS, VARMA, Wilson–Cowan, and others). Only models that pass validation are included in the final benchmark.
2. **Fixed protocol** — For every model, data is generated with the same protocol:
  - **Training length**: fixed number of training observations (e.g. 5000) and forecast length (e.g. 300).
  - **Burn-in**: a fixed number of burn-in steps (e.g. 1000) so that outputs are from the attractor rather than transients.
  - **Same time step and numerical method** per generator (defined in each `DataGenerator`).
3. **Three noise levels per model** — Each benchmark model is generated in three variants:
  - **Deterministic** — no process noise (or generator default).
  - **Low stochastic** — process noise scaled by a fixed fraction (e.g. 0.1) of the empirical standard deviation of the training data from the deterministic run.
  - **High stochastic** — same idea with a larger fraction (e.g. 0.25).  
   Outputs are written to subfolders (e.g. `Deterministic/`, `LowStochastic/`, `HighStochastic/`) so that methods can be evaluated across noise regimes.
4. **Consistent data format and metadata** — Every dataset is exported as JSON via `ControlVsRespData.to_json()`, which includes:
  - Train and prior states/times, forecast states/times, and causal response (intervention) states/times.
  - Exogenous indices for observational and interventional variables.
  - A **model description** string (e.g. from `make_description(dg)`) for reproducibility.
  - Schema-oriented metadata (variable names, types, descriptions) so that the benchmark is self-describing.
5. **Validation and filtering** — The notebook applies simple checks during generation:
  - **Response vs forecast error**: generators where the intervention-response error is too close to the forecast error (e.g. < 1.1×) are flagged as “bad” (intervention may not be sufficiently distinct).
  - **Explosions**: runs that produce NaNs or very large values are recorded (e.g. for sensitivity to noise or step size).  
   This keeps the benchmark to models that are well-behaved under the chosen protocol and noise levels.
6. **Reproducibility** — The notebook pins an interfere commit hash and uses fixed seeds in the package (`DataGenerator` and `control_vs_resp`), so the same benchmark can be regenerated from the same code and parameters.

Together, these choices yield a **reproducible, multi-model, multi-noise benchmark** for evaluating forecasters and causal predictors on the control-vs-response problem, with a single script (the notebook) driving generation for all models and variants.

## Quick usage

```python
import interfere_experiments as ie
from interfere_experiments.data_generators import Lorenz1, LotkaVoltera1

# Generate control-vs-response data from a predefined generator
dg = Lorenz1()
data = dg.generate_data(num_train_obs=500, num_forecast_obs=100, num_burn_in_states=200)

# data is a ControlVsRespData instance: train_states, forecast_states, interv_states, etc.
# Export to JSON (as in the benchmark)
data.to_json("path/to/lorenz1.json", model_description="Lorenz system, default params.")

# Load a benchmark-style JSON
cvr = ie.control_vs_resp.load_cvr_json("path/to/AttractingFixedPoint4D.json")
```

## Repo structure (relevant to the benchmark)

- `**interfere_experiments/**` — Package: `data_generators`, `control_vs_resp`, `quick_models`.
- `**notebooks/GeneratingInterfereBenchmark1.1.1.ipynb**` — Systematic construction of the Interfere Benchmark (1.1.1): defines `BENCHMARK_MODELS`, loops over models and noise levels, validates, and writes JSON.
- `**experiments/**` — Example experiment scripts that load benchmark JSON (e.g. via `load_cvr_json`) and run forecasters.

## Installation

From the repo root (with pip and git):

```bash
pip install -e .
```

Optional extras: `pip install -e ".[dev,notebooks]"` for tests and Jupyter. See `pyproject.toml` for dependencies (interfere and optional dev/notebooks).