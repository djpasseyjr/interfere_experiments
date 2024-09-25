import pickle as pkl


import interfere
import interfere_experiments as ie
from interfere_experiments.quick_models import gut_check_coupled_logistic, gut_check_belozyorov
import matplotlib.pyplot as plt
import numpy as np
import optuna


SAVE_DIR = "/work/users/d/j/djpassey/interfere_exp6.0"

SEED = 11
RNG = np.random.default_rng(SEED)
METHODS = [
    interfere.methods.LTSFLinearForecaster,
    interfere.methods.VAR,
    interfere.methods.LSTM,
    interfere.methods.NHITS,
    interfere.methods.ResComp,
    interfere.methods.SINDY,
]
MODEL_NAMES = model_names = ["cml", "belozy"]
TRIALS_PER_METHOD = 25

# Make arguments for models.
dynamic_model_args = {
    "cml": {
            # Two cycles and isolated node
            "adjacency_matrix": np.array([
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]),
            "eps": 0.9,
            "logistic_param": 3.72,
            "measurement_noise_std": None
    },

    "belozy": dict(
        mu=1.81, measurement_noise_std = None
    )
}


# Make arguments for the control vs response objective.

cml_train_prior_states = RNG.random((5, 10))
belozy_train_prior_states = -0.1 * np.ones((2, 3))

obj_dynamics_args = {
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
imgs = {m.__name__: {} for m in METHODS}


for sigma in np.linspace(0, 0.5, 11):

    for model_type, model_key in zip(
        [
            interfere.dynamics.coupled_logistic_map,
            interfere.dynamics.Belozyorov3DQuad
        ],
        ["cml", "belozy"]
    ):
        model = model_type(
            **dynamic_model_args[model_key],
            sigma=sigma,
        )

        model_name = f"{model_type.__name__}(sigma={sigma:.2f})"

        for method in METHODS:
            objective = ie.control_vs_resp.CVROptunaObjective(
                model=model,
                method_type=method,
                **obj_dynamics_args[model_key],
                rng=RNG,  
            )
            study = optuna.create_study(
                study_name = f"{method.__name__} predicting {model_name}",
                directions=objective.metric_directions
            )
            study.optimize(objective, n_trials=TRIALS_PER_METHOD)

            # Save data
            studies[method.__name__][model_name] = study
            imgs[method.__name__][model_name] = {
                k: np.array(v) for k, v in objective.trial_imgs.items()}

            pkl.dump(
                {"studies": studies, "imgs": imgs},
                open(SAVE_DIR + 'stochastic_sweep_studies.pkl', 'wb')
            )