import pickle as pkl
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


import interfere
import interfere.methods
import interfere_experiments as ie
from interfere_experiments.quick_models import gut_check_coupled_logistic, gut_check_belozyorov
import numpy as np
import optuna


SAVE_DIR = "/work/users/d/j/djpassey/interfere_exp5.2/"
with open(Path(__file__).parent / 'params.pkl', 'rb') as f:
    GUT_CHECK_PARAMS = pkl.load(f)
SEED = 11
RNG = np.random.default_rng(SEED)
METHODS = [
    interfere.methods.LTSF,
    interfere.methods.VAR,
    interfere.methods.LSTM,
    interfere.methods.NHITS,
    interfere.methods.ResComp,
    interfere.methods.SINDY,
]
PREDICTOR_METHODS = METHODS + [interfere.methods.AverageMethod] 

MODEL_NAMES = model_names = ["cml", "belozy"]
TRIALS_PER_METHOD = 25

# Create CML Training Data.

dim = 10
dt = 1
window = 10
num_train_obs=350

cml_train_prior_states = RNG.random((window, dim))
cml_train_prior_t = np.arange(-window * dt + 1, 1, dt)
cml_train_t = np.arange(0, num_train_obs * dt, dt)

# Simulate training data.
cml_train_states = gut_check_coupled_logistic().simulate(
    cml_train_t, 
    prior_states=cml_train_prior_states,
    prior_t=cml_train_prior_t,
    rng=RNG
)


# Create Belozyorov Training Data.

dim = 3
dt = 0.05
window = 10
num_train_obs=350

belozy_train_prior_states = -0.1 * np.ones((2, dim))
belozy_train_prior_t = np.array([-dt, 0])
belozy_train_t = np.arange(0, num_train_obs * dt, dt)

# Simulate training data.
belozy_train_states = gut_check_belozyorov().simulate(
    belozy_train_t, 
    prior_states=belozy_train_prior_states,
    prior_t=belozy_train_prior_t,
    rng=RNG,
)

# Make arguments for the control vs response o
dynamics_args = {
    "cml": dict(
        num_train_obs=150,
        num_forecast_obs=50,
        timestep=1.0,
        intervention=interfere.PerfectIntervention(0, 1.0),
        train_prior_states=cml_train_prior_states
    ),
    "belozy": dict(
        num_train_obs=150,
        num_forecast_obs=50,
        timestep=0.05,
        intervention=interfere.PerfectIntervention(2, 0.2),
        train_prior_states=belozy_train_prior_states
    )
}


# Optimization loop.
studies = {m.__name__: {} for m in METHODS}
data = {m.__name__: {} for m in METHODS}

for train_t, train_states, base_dyn_name in zip(
    [cml_train_t, belozy_train_t],
    [cml_train_states, belozy_train_states],
    ["cml", "belozy"]
):
    
    # Fit Methods and Create GenerativeForecasters.    
    gen_fcast = []
    for method_type in METHODS:
        params = GUT_CHECK_PARAMS[
            method_type.__name__][base_dyn_name]["forecast_rmse"]
        method = method_type(**params)
        method.fit(train_t, train_states)
        gen_fcast.append(
            interfere.dynamics.GenerativeForecaster(method)
        )

    for gen_forecaster in gen_fcast:

        model_name = (
            "GenForecast["
            f"{base_dyn_name}, "
            f"{type(gen_forecaster.fitted_method).__name__}"
            "]"
        )

        for method in PREDICTOR_METHODS:
            objective = ie.control_vs_resp.CVROptunaObjective(
                model=gen_forecaster,
                method_type=method,
                **dynamics_args[base_dyn_name],
                store_plots=False,
                rng=RNG,  
            )
            study = optuna.create_study(
                study_name = f"{method.__name__} predicting {model_name}",
                directions=objective.metric_directions
            )
            study.optimize(objective, n_trials=TRIALS_PER_METHOD)

            # Save data
            studies[method.__name__][model_name] = study
            data[method.__name__][model_name] = {
                "target": objective.data,
                "trials": objective.trial_preds,
            }

            pkl.dump(
                {"studies": studies, "data": data},
                open(SAVE_DIR + 'generative_forecaster_studies.pkl', 'wb')
            )
