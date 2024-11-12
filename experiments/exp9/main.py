from datetime import datetime, timedelta
import os
import pickle
import json

import interfere.methods
import interfere_experiments as ie
import numpy as np
import optuna

DATA_DIR = "/nas/longleaf/home/djpassey/InterfereBenchmark0.0.1/"
SAVE_DIR = "/work/users/d/j/djpassey/interfere_exp9.0/"
METHODS = [
    interfere.methods.VAR,
    interfere.methods.SINDY,
    interfere.methods.ResComp,
    interfere.methods.LTSF,
    interfere.methods.NHITS,
    interfere.methods.LSTM
]
OPTUNA_TRIALS_DICT = {
    interfere.methods.VAR: 51,
    interfere.methods.SINDY: 101,
    interfere.methods.ResComp: 201,
    interfere.methods.LTSF: 101,
    interfere.methods.NHITS: 101,
    interfere.methods.LSTM: 51,
}

TRAIN_WINDOW_PERCENT = 0.75
NUM_FOLDS = 3
NUM_VAL_PRIOR_STATES = 50
METRIC = interfere.metrics.RootMeanSquaredScaledErrorOverAvgMethod()
DATA_IDX = 1
NUM_TRAIN_OBS_LIST = [100, 200, 500, 1000, 2000, 3650]
NUM_TEST_OBS = 100

data_files = os.listdir(DATA_DIR)
data_name = data_files[DATA_IDX].split(".")[0]
file = DATA_DIR + data_files[DATA_IDX]

with open(file, "r") as f:
    data = json.load(f)

score_file = SAVE_DIR + data_name + "_scores.pkl"
pred_file = SAVE_DIR + data_name + "_preds.pkl"

all_train_states = np.array(data["train_states"])
all_train_times = np.array(data["train_times"])


all_forecast_states = np.array(data["forecast_states"])
all_forecast_times = np.array(data["forecast_times"])
test_states = all_forecast_states[:NUM_TEST_OBS]
test_times = all_forecast_times[:NUM_TEST_OBS]

scores = {
    m.__name__: {
        obs: {
            "score": None,
            "duration": None,
            "n_trials": None,
            "pred_file_size_est": None,
        }
        for obs in NUM_TRAIN_OBS_LIST
    } for m in METHODS
}
predictions = {
    m.__name__: {
        obs: {
            "best_method_preds": None,
            "test_pred": None
        } for obs in NUM_TRAIN_OBS_LIST
    } for m in METHODS
}

for method_type in METHODS:
    for num_train_obs in NUM_TRAIN_OBS_LIST:

        start_time = datetime.now()

        train_states = all_train_states[-num_train_obs:]
        train_times = all_train_times[-num_train_obs:]

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
            raise_errors=False
        )

        study = optuna.create_study(
            study_name=method_type.__name__ + " on " + data_name)
        n_trials = OPTUNA_TRIALS_DICT[method_type]
        study.optimize(objv, n_trials=n_trials)

        best_params = study.best_params

        best_method = method_type(**best_params).fit(train_times, train_states)
        pred_states = best_method.predict(
            test_times,
            prior_endog_states=train_states,
            prior_t = train_times
        )
        score = METRIC(train_states, test_states, pred_states, [])

        end_time = datetime.now()
        duration = str(end_time - start_time)
        pred_sz = len(pickle.dumps(objv.trial_results[study.best_trial.number]))
        scores[method_type.__name__][num_train_obs]["score"] = score
        scores[method_type.__name__][num_train_obs]["duration"] = duration
        scores[method_type.__name__][num_train_obs]["n_trials"] = n_trials
        scores[method_type.__name__][num_train_obs][
            "pred_file_size_est"] = pred_sz

        predictions[method_type.__name__][num_train_obs][
            "best_method_preds"] = objv.trial_results[study.best_trial.number]
        predictions[method_type.__name__][num_train_obs][
            "test_pred"] = pred_states
        
        with open(score_file, "wb") as sf:
            pickle.dump(scores, sf)


with open(pred_file, "wb") as pf:
    pickle.dump(predictions, pf)