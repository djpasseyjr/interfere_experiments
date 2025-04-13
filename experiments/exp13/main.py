from datetime import datetime
import os
from pathlib import Path
import pickle
import sys
import traceback

import interfere.methods
import interfere_experiments as ie
import numpy as np
import optuna

# Collect script arguments
DATA_FILE = sys.argv[1]

METHOD_GROUP = None
if len(sys.argv) > 2:
    METHOD_GROUP = sys.argv[2]


SAVE_DIR = "/work/users/d/j/djpassey/interfere_exp13.1/"
ALL_METHODS = [
    interfere.methods.AverageMethod,
    interfere.methods.VAR,
    interfere.methods.SINDY,
    interfere.methods.ResComp,
    interfere.methods.ARIMA,
    interfere.methods.LSTM,
]
FAST_METHODS = [
    interfere.methods.AverageMethod,
    interfere.methods.VAR,
    interfere.methods.SINDY,
    interfere.methods.ResComp,
]

METHOD_GROUPS = {
    None: ALL_METHODS,
    "FAST": FAST_METHODS,
    "ARIMA": [interfere.methods.ARIMA],
    "NHITS": [interfere.methods.NHITS],
    "LSTM": [interfere.methods.LSTM]
}

METHODS = METHOD_GROUPS[METHOD_GROUP]

OPTUNA_TRIALS_DICT = {
    interfere.methods.AverageMethod: 1,
    interfere.methods.VAR: 51,
    interfere.methods.SINDY: 101,
    interfere.methods.ResComp: 201,
    interfere.methods.ARIMA: 51,
    interfere.methods.NHITS: 101,
    interfere.methods.LSTM: 101,
}

TRAIN_WINDOW_PERCENT = 0.5
NUM_FOLDS = 3
NUM_VAL_PRIOR_STATES = 25
METRIC = interfere.metrics.RootMeanStandardizedSquaredError()
NUM_TRAIN_OBS_LIST = [100, 200, 500, 1000, 2000, 3000, 4000, 5000]

data_name = os.path.basename(DATA_FILE).split(".")[0]
cvr_data = ie.control_vs_resp.load_cvr_json(DATA_FILE)

f = Path(DATA_FILE)
deterministic_data_file = f.parent.with_name("Deterministic") / f.name

target_cvr_data = ie.control_vs_resp.load_cvr_json(deterministic_data_file)

score_file = SAVE_DIR + data_name + f"_scores[{str(METHOD_GROUP)}].pkl"
pred_file = SAVE_DIR + data_name + f"_preds[{str(METHOD_GROUP)}].pkl"

# Initialize empty results dictionaries
score_df_cols = [
     "Dynamics", "Method", "Obs", "ForecastError", "CausalError", 
     "Duration", "Trials", "Size", "Exceptions"
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
            train_states = cvr_data.train_states[-num_train_obs:]
            train_times = cvr_data.train_t[-num_train_obs:]

            # Build a cross validation objective to optimize.
            objv = interfere.cross_validation.CrossValObjective(
                method_type,
                train_states,
                train_times,
                TRAIN_WINDOW_PERCENT,
                NUM_FOLDS,
                val_scheme="forecast",
                num_val_prior_states=NUM_VAL_PRIOR_STATES,
                metric=METRIC,
                store_preds=True,
                raise_errors=False,
                exog_idxs=cvr_data.do_intervention.intervened_idxs,
            )

            study = optuna.create_study(
                study_name=method_type.__name__ + " on " + data_name + f"({num_train_obs})")
            n_trials = OPTUNA_TRIALS_DICT[method_type]
            study.optimize(objv, n_trials=n_trials)

            # Try scoring the best method on hold out test set.
            try:
                best_params = study.best_params

                # Forecast.
                obs_method = method_type(**best_params)
                obs_method.fit(
                    train_times, 
                    *target_cvr_data.obs_intervention.split_exog(train_states)
                ) 
                forecast_pred = obs_method.simulate(
                    target_cvr_data.forecast_t,
                    prior_states=target_cvr_data.train_states,
                    prior_t=target_cvr_data.train_t,
                    intervention=target_cvr_data.obs_intervention,
                )

                # Forecast error.
                fcast_error = METRIC(
                    target_cvr_data.train_states[:, [target_cvr_data.target_idx]],
                    target_cvr_data.forecast_states[:, [target_cvr_data.target_idx]],
                    forecast_pred[:, [target_cvr_data.target_idx]],
                    []
                )

            except Exception as e:
                fcast_error = np.nan
                forecast_pred = None
                errors += f"\n\nERROR {e}" + str(traceback.format_exc())


            # Causal prediction.
            try:
                best_params = study.best_params

                do_method = method_type(**best_params)
                do_method.fit(
                    train_times, 
                    *cvr_data.do_intervention.split_exog(train_states)
                )
                causal_pred = do_method.simulate(
                    target_cvr_data.forecast_t,
                    prior_states=target_cvr_data.train_states,
                    prior_t=target_cvr_data.train_t,
                    intervention=target_cvr_data.do_intervention,
                )

                # Causal prediction error.
                causal_error = METRIC(
                    target_cvr_data.train_states[:, [target_cvr_data.target_idx]],
                    target_cvr_data.interv_states[:, [target_cvr_data.target_idx]],
                    causal_pred[:, [target_cvr_data.target_idx]],
                    []
                )

            except Exception as e:
                causal_error = np.nan
                causal_pred = None
                errors += f"\n\nERROR {e}" + str(traceback.format_exc())

            
            try:
                best_trial_idx = study.best_trial.number
            except ValueError as e:
                print(
                    "Error finding best trial: "
                    f"\n\n{e}\n\n{traceback.format_exc()}"
                )
                best_trial_idx = 0


            end_time = datetime.now()
            duration = str(end_time - start_time)
            pred_sz = len(pickle.dumps(objv.trial_results[best_trial_idx]))

            # Save scores.
            score_array.append([
                data_name, method_type.__name__, num_train_obs, fcast_error, causal_error,
                duration, n_trials, pred_sz, errors
            ])

            # Save predictions.
            predictions[method_type.__name__][num_train_obs][
                "train_pred"] = objv.trial_results[best_trial_idx]
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

