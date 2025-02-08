from datetime import datetime
import json
import os
import pickle
import sys
import traceback

import interfere.methods
import interfere_experiments as ie
import numpy as np
import optuna
import scipy.interpolate

DATA_FILE = sys.argv[1]
SAVE_DIR = "/work/users/d/j/djpassey/interfere_exp12.0/"
METHODS = [
    interfere.methods.AverageMethod,
    interfere.methods.VAR,
    interfere.methods.SINDY,
    interfere.methods.ResComp,
    interfere.methods.ARIMA,
    interfere.methods.LSTM,
    interfere.methods.NHITS,

]
OPTUNA_TRIALS_DICT = {
    interfere.methods.AverageMethod: 2, #1,
    interfere.methods.VAR: 2, #51,
    interfere.methods.SINDY: 2, #101,
    interfere.methods.ResComp: 2, #201,
    interfere.methods.ARIMA: 2, #51,
    interfere.methods.NHITS: 2, #101,
    interfere.methods.LSTM: 2, #101,
}

TRAIN_WINDOW_PERCENT = 0.5
NUM_FOLDS = 3
NUM_VAL_PRIOR_STATES = 25
METRIC = interfere.metrics.RootMeanStandardizedSquaredError()
NUM_TRAIN_OBS_LIST = [100]#, 200, 500, 1000, 2000, 3000, 4000, 5000]

data_name = os.path.basename(DATA_FILE).split(".")[0]

with open(DATA_FILE, "r") as f:
    data = json.load(f)

score_file = SAVE_DIR + data_name + "_scores.pkl"
pred_file = SAVE_DIR + data_name + "_preds.pkl"

# Load train data.
all_train_states = np.array(data["train_states"])
all_train_times = np.array(data["train_times"])
train_exog_idxs = data["train_exog_idxs"]

# Load forecast data.
all_forecast_states = np.array(data["forecast_states"])
all_forecast_times = np.array(data["forecast_times"])

# Load intervention response data.
all_causal_resp_states = np.array(data["causal_resp_states"])
all_causal_resp_times = np.array(data["causal_resp_times"])
causal_exog_idxs = data["causal_resp_exog_idxs"]
target_idx = data["target_idx"]

# Initialize observational intervention.
obs_interv = interfere.SignalIntervention(
    train_exog_idxs, 
    # Interpolate the exogenous signals from the training and forecasting data.
    [
        scipy.interpolate.interp1d(
            x = np.hstack([all_train_times[:-1], all_forecast_times]),
            y = np.hstack([all_train_states[:-1, i], all_forecast_states[:, i]])
        )
        for i in train_exog_idxs
    ]
)

# Initialize causal intervention.
do_interv = interfere.SignalIntervention(
    causal_exog_idxs, 
    # Interpolate the exogenous signals from the train and causal response data.
    [
        scipy.interpolate.interp1d(
            x = np.hstack([all_train_times[:-1], all_forecast_times]),
            y = np.hstack([
                all_train_states[:-1, i], 
                all_causal_resp_states[:, i]
            ])
        )
        for i in causal_exog_idxs
    ]
)

# Initialize empty results dictionaries
score_df_cols = [
     "Method", "Obs", "Forecast Error", "Causal Error", 
     "Duration", "Trials", "Size", "Errors"
]
score_array = []

predictions = {
    m.__name__: {
        obs: {
            "forecast_pred": None,
            "causal_pred": None,
            "train_pred": None
        } for obs in NUM_TRAIN_OBS_LIST
    } for m in METHODS
}

try:
    # Main experiment loop.
    for method_type in METHODS:
        for num_train_obs in NUM_TRAIN_OBS_LIST:

            errors = "Errors:"
            start_time = datetime.now()

            # Take the train data prior to the forecast.
            train_states = all_train_states[-num_train_obs:]
            train_times = all_train_times[-num_train_obs:]

            # Build a cross validation objective to optimize.
            objv = ie.cross_validation.CVRCrossValObjective(
                method_type,
                train_states,
                train_times,
                TRAIN_WINDOW_PERCENT,
                NUM_FOLDS,
                val_scheme="forecast",
                num_val_prior_states=NUM_VAL_PRIOR_STATES,
                metric=METRIC,
                store_preds=True,
                raise_errors=True,
                exog_idxs=train_exog_idxs,
            )

            study = optuna.create_study(
                study_name=method_type.__name__ + " on " + data_name)
            n_trials = OPTUNA_TRIALS_DICT[method_type]
            study.optimize(objv, n_trials=n_trials)

            # Try scoring the best method on hold out test set.
            try:
                best_params = study.best_params

                # Forecast.
                obs_method = method_type(**best_params)
                obs_method.fit(
                    train_times, *obs_interv.split_exog(train_states)) 
                forecast_pred = obs_method.simulate(
                    all_forecast_times,
                    prior_states=train_states,
                    prior_t=train_times,
                    intervention=obs_interv,
                )

                # Forecast error.
                fcast_error = METRIC(
                    all_train_states[:, [target_idx]],
                    all_forecast_states[:, [target_idx]],
                    forecast_pred[:, [target_idx]],
                    []
                )

            except Exception as e:
                fcast_error = np.nan
                forecast_pred = None
                errors += f"\n\nERROR {e}" + str(traceback.format_exc())


            # Causal prediction.
            try:
                do_method = method_type(**best_params)
                do_method.fit(train_times, *do_interv.split_exog(train_states))
                causal_pred = do_method.simulate(
                    all_forecast_times,
                    prior_states=train_states,
                    prior_t=train_times,
                    intervention=do_interv,
                )

                # Causal prediction error.
                causal_error = METRIC(
                    all_train_states[:, [target_idx]],
                    all_forecast_states[:, [target_idx]],
                    causal_pred[:, [target_idx]],
                    []
                )

            except Exception as e:
                causal_error = np.nan
                causal_pred = None
                errors += f"\n\nERROR {e}" + str(traceback.format_exc())

            end_time = datetime.now()
            duration = str(end_time - start_time)
            pred_sz = len(pickle.dumps(objv.trial_results[study.best_trial.number]))

            # Save scores.
            score_array.append([
                 method_type.__name__, num_train_obs, fcast_error, causal_error,
                 duration, n_trials, pred_sz, errors
            ])

            # Save predictions.
            predictions[method_type.__name__][num_train_obs][
                "train_pred"] = objv.trial_results[study.best_trial.number]
            predictions[method_type.__name__][num_train_obs][
                "forecast_pred"] = forecast_pred
            predictions[method_type.__name__][num_train_obs][
                "causal_pred"] = causal_pred
            
            with open(score_file, "wb") as sf:
                pickle.dump(
                    {"cols": score_df_cols, "score_array": score_array}, sf)
                
            with open(pred_file, "wb") as pf:
                pickle.dump(predictions, pf)

except Exception as e:
    print("Experiment ended because of exception.")
    print(type(e).__name__, ": ", e)
    print(traceback.format_exc())

