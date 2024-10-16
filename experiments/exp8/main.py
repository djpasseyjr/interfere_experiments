import pickle as pkl

import interfere
import interfere.methods
import interfere_experiments as ie
import numpy as np
import optuna


SAVE_DIR = "/work/users/d/j/djpassey/interfere_exp8.0/"

SEED = 11
RNG = np.random.default_rng(SEED)
METHODS = [
    interfere.methods.AverageMethod,
    interfere.methods.VAR,
    interfere.methods.ResComp,
    interfere.methods.SINDY,
    interfere.methods.LTSF,
    interfere.methods.LSTM,
    interfere.methods.NHITS,
]
MODELS = [
    ie.quick_models.gut_check_coupled_logistic(),
    ie.quick_models.gut_check_belozyorov()
]
MODEL_NAMES = ["cml", "belozy"]
MODEL_INTERV_IDX = {
      "cml": 1,
      "belozy": 2,
}
MODEL_INTERV_CONSTS = {
      "cml": [x/10 for x in range(10)],
      "belozy": [-x/10 for x in range(10)]
}
# Make arguments for the control vs response objective.
MODEL_ARGS = {
    "cml": dict(
        timestep=1.0,
        train_prior_states=RNG.random((5, 10))
    ),
    "belozy": dict(
        timestep=0.05,
        train_prior_states=RNG.random((5, 3))
    )
}

TRIALS_PER_METHOD = 25
NUM_TRAIN_OBS = 300
NUM_FORECAST_OBS = 50


# Storage dicts.
studies = {
    model: {
        method.__name__: {}
        for method in METHODS
    }
    for model in MODEL_NAMES
}
data = {
    model: {
        method.__name__: {}
        for method in METHODS
    }
    for model in MODEL_NAMES
}

# Optimization loop.
for model, model_name in zip(
  MODELS,
  MODEL_NAMES
):
    for c in MODEL_INTERV_CONSTS[model_name]:

        intervention = interfere.PerfectIntervention(
            MODEL_INTERV_IDX[model_name], c)

        for method in METHODS:

            # Wrap gen forecaster in a control vs response objective func.
            objective = ie.control_vs_resp.CVROptunaObjective(
                model=model,
                method_type=method,
                num_train_obs=NUM_TRAIN_OBS,
                num_forecast_obs=NUM_FORECAST_OBS,
                intervention=intervention,
                store_plots=False,
                rng=RNG,  
                **MODEL_ARGS[model_name],
            )

            study = optuna.create_study(
                study_name = f"{method.__name__} predicting {model_name}",
                directions=objective.metric_directions
            )
            # Run hyper parameter optimization.
            study.optimize(objective, n_trials=TRIALS_PER_METHOD)

            # Save data
            studies[model_name][method.__name__][c] = study
            data[model_name][method.__name__][c] = {
                "target": objective.data,
                "trials": objective.trial_preds,
                "errors": objective.trial_error_log
            }

            pkl.dump(
                {"studies": studies, "data": data},
                open(SAVE_DIR + 'intervention_sweep.pkl', 'wb')
            )