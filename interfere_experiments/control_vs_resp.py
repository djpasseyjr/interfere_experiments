from dataclasses import dataclass
import datetime
import json
from pathlib import Path
from tempfile import TemporaryFile
import traceback
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

import PIL.Image

import interfere
from interfere.interventions import ExogIntervention, IdentityIntervention
from interfere.metrics import RootMeanStandardizedSquaredError as RMSSE
import matplotlib.pyplot as plt
import numpy as np
from optuna.trial import Trial
import PIL

DEFAULT_RNG = np.random.default_rng()


@dataclass
class ControlVsRespData:
    """A dataclass to organize the control vs response forecasting problem.

    Args:
        train_prior_t (np.ndarray): An array of time points with shape (p,)
            used as historic obs when generating the training data. Can be used
            to try to reproduce the training data if the algorithm has lags
            or time delays, etc.. 
        train_prior_states (np.ndarray): The states of the system used as
            historic obs when generating the training data. An array with shape
            (p, n) where rows are observations and columns are variables.
        obs_intervention (ExogIntervention): Intervention controlling 
            exogenous states during the observational stages (training and 
            forecasting).
        train_t (np.ndarray): An array of time points with shape (m,)
            corresponding to the training data.
        train_states (np.ndarray): An array of training states of the system
            with shape (m, n). Rows are observations and columns are
            variables.
        forecast_t (np.ndarray): An array of time points with shape (k,)
            corresponding to the forecast data.
        forecast_states (np.ndarray): An array of states of the system to
            forecast with shape (k, n). Rows are observations and columns
            are variables.
        do_intervention (ExogIntervention): Intervention controlling   
            observational exogenous states (obs_intervention) AND intervened 
            states.
        interv_states (np.ndarray): An array containing the intervention
            response with shape (k, n). Rows are observations and columns
            are variables. Each row corresponds to the times in
            `forecast_t`.
            
    Description:

        The control v.s. response forecasting problem involves 8 components
        and has 3 stages. 

        Stage 1: Training. During this stage, a forecaster fits to the data
        in `train_states`, an array where each row is an observation that
        occurred at the corresponding time in `train_t`.

        Stage 2: Classical forecasting. During this stage, the forecaster
        makes a prediction about what will happen in the times contained in
        `forecast_t` which typically correspond to the time immidiately
        following the training data.

        Stage 3: Intervention response forecasting. This stage attempts to
        predict the time series contained in `interv_states` which
        correspond to the response of the data generating process to an
        exogenous intervention. The intervention is modeled as taking
        exogenous control of one or more of the endogenous states from the
        training data. 
    """
    train_prior_t: np.ndarray
    train_prior_states: np.ndarray
    obs_intervention: ExogIntervention
    train_t: np.ndarray
    train_states: np.ndarray
    forecast_t: np.ndarray
    forecast_states: np.ndarray
    do_intervention: ExogIntervention
    interv_states: np.ndarray


    def __post_init__(self):
        """Validates that attribute arrays are the correct shape.

        Raises:
            ValueError: If len of time arrays is different than the number of 
                rows in the corresponding states array. Error message contains 
                the mismatched shapes.
            ValueError: If any of the state arrays have different numbers of 
                columns.
        """
        # Check that time and state arrays match.
        if self.train_prior_t.shape[0] != self.train_prior_states.shape[0]:
            raise ValueError(
                "train_prior_t and train_prior_states must have the same number of rows."
                f"\ntrain_prior_t.shape = {self.train_prior_t.shape}, "
                f"\ntrain_prior_states.shape = {self.train_prior_states.shape}"
            )

        if self.train_t.shape[0] != self.train_states.shape[0]:
            raise ValueError(
                "train_t and train_states must have the same number of rows."
                f"\ntrain_t.shape = {self.train_t.shape}, "
                f"\ntrain_states.shape = {self.train_states.shape}"
            )

        if self.forecast_t.shape[0] != self.forecast_states.shape[0]:
            raise ValueError(
                "forecast_t and forecast_states must have the same number of rows."
                f"\nforecast_t.shape = {self.forecast_t.shape}, "
                f"\nforecast_states.shape = {self.forecast_states.shape}"
            )

        if self.interv_states.shape[0] != self.forecast_t.shape[0]:
            raise ValueError(
                "interv_states and forecast_t must have the same number of rows."
                f"\ninterv_states.shape = {self.interv_states.shape}, "
                f"\nforecast_t.shape = {self.forecast_t.shape}"
            )

        
        # Check that all state arrays have the same number of columns.
        state_arrays = [
            self.train_prior_states, self.train_states, 
            self.forecast_states, self.interv_states
        ]

        for state_array in state_arrays:
            if state_array.shape[1] != state_arrays[0].shape[1]:
                raise ValueError(
                    "All state arrays must have the same number of columns."
                    f"\ntrain_prior_states.shape = {self.train_prior_states.shape}"
                    f"\ntrain_states.shape = {self.train_states.shape}"
                    f"\nforecast_states.shape = {self.forecast_states.shape}"
                    f"\ninterv_states.shape = {self.interv_states.shape}"
                )
    
        obs_exog_idxs = set(self.obs_intervention.intervened_idxs)
        do_exog_idxs = set(self.do_intervention.intervened_idxs)
        if (
            not obs_exog_idxs.issubset(do_exog_idxs)
            or
            do_exog_idxs.issubset(obs_exog_idxs)
        ):
            raise ValueError((
                "Observational intervention exogenous variables must be a "
                "proper subset of causal intervention exogenous variables."
                f"\nObs Exog: {list(obs_exog_idxs)}"
                f"\nCausal Exog: {list(do_exog_idxs)}"
            ))
            
    def __eq__(self, other):
        """Compare equality across two ControlVsRespData classes.
        """
        if not isinstance(other, ControlVsRespData):
            return False
        
        equal = True
        equal = equal and np.all(self.train_prior_t == other.train_prior_t)
        equal = equal and np.all(
            self.train_prior_states == other.train_prior_states)
        equal = equal and np.all(self.train_t == other.train_t)
        equal = equal and np.all(self.train_states == other.train_states)
        equal = equal and np.all(self.forecast_t == other.forecast_t)
        equal = equal and np.all(self.forecast_states == other.forecast_states)
        equal = equal and np.all(self.interv_states == other.interv_states)
        equal = equal and (self.obs_intervention == other.obs_intervention)
        equal = equal and (self.do_intervention == other.do_intervention)

        return equal
    

    def json_metadata(self):
        """Generates metadata for JSON export."""
        return {
                "overview": """This dataset contains simulated timeseries data.It contains observational data followed by a control scenario (classical forecast problem) and an intervention response scenario (causal prediction problem). The name, type and a description of each variable are contained under ['metadata']['variables']. Typing is specified according to the python `typing` library. The date_created field is formatted as D/M/Y H:M:S.""",
                "date_created": datetime.datetime.now().strftime(
                    "%d/%m/%Y %H:%M:%S"),
                "variables": [
                    {
                        "name": "model_description",
                        "type": "str",
                        "description": "A string of text describing the dynamic model that underlies the data generating process. May include equations, references, parameter values and parameter descriptions."
                    },
                    {
                        "name": "train_states", 
                        "type": "List[List[float]]", 
                        "description": "An mxn list of lists. Rows are observations, columns are variables. Observational data that occurs directly prior to forecast_states and intervention_states."
                    },
                    {
                        "name": "train_times", 
                        "type": "List[float]",
                        "description": "A list with length m. Contains the times corresponding to each observation in (row of) train_states."
                    },
                    {
                        "name": "train_exog_idxs", 
                        "type": "List[int]",
                        "description": "The zero based column index that corresponds to the exgoenous variables columns train_states. Always the same as forecast_exog_idxs."
                    },
                    {
                        "name": "forecast_states",
                        "type": "List[List[float]]",
                        "description": "An kxn list of lists. Rows are observations, columns are variables. Data that occurs directly after the training data."
                    },
                    {
                        "name": "forecast_times", 
                        "type": "List[float]",
                        "description": "A list with length k. Contains the times corresponding to each observation in (row of) forecast_states."
                    },
                    {
                        "name": "forecast_exog_idxs", 
                        "type": "List[int]",
                        "description": "The zero based column index that corresponds to the exgoenous variables columns forecast_states. Always the same as train_exog_idxs."
                    },
                    {
                        "name": "causal_resp_states",
                        "type": "List[List[float]]",
                        "description": "An kxn list of lists. Rows are observations, columns are variables. Contains data that occurs directly after the training data in response to exogenous control of a formerly endogenous variable."
                    },
                    {
                        "name": "causal_resp_times", 
                        "type": "List[float]",
                        "description": "A list with length k. Contains the times corresponding to each observation in (row of) causal_resp_states."
                    },
                    {
                        "name": "causal_resp_exog_idxs", 
                        "type": "List[int]",
                        "description": "The zero based column index that corresponds to the exogenous variables that were manipulated to produce causal_resp_states. Contains all indexes in train_exog_idxs and always includes at least one additional index corresponding to a formerly endogenous variable."
                    }
                ]
            }
    

    def to_json(self, file_path: Union[str, Path], model_description: str = ""):
        """Export ControlVsRespData to JSON.

        Args:
            model_description (str): Optional description of the model that     
                generated the data.
        """
        # Build dictionary of data.
        cvr_dict = {
            "metadata": self.json_metadata(),
            "model_description": model_description,
            "train_states": self.train_states.tolist(),
            "train_times": self.train_t.tolist(),
            "train_exog_idxs": self.obs_intervention.intervened_idxs,
            "forecast_states": self.forecast_states.tolist(),
            "forecast_times": self.forecast_t.tolist(),
            "forecast_exog_idxs": self.obs_intervention.intervened_idxs,
            "causal_resp_states": self.interv_states.tolist(),
            "causal_resp_times": self.forecast_t.tolist(),
            "causal_resp_exog_idxs": self.do_intervention.intervened_idxs,
        }
        with open(file_path, "w") as f:
            json.dump(cvr_dict, f)
        

def generate_data(
    model: interfere.dynamics.base.DynamicModel,
    num_train_obs: int,
    num_forecast_obs: int,
    timestep: float,
    do_intervention: ExogIntervention,
    obs_intervention: ExogIntervention = interfere.interventions.IdentityIntervention(),
    train_prior_states: Optional[np.ndarray] = None,
    lags: Optional[int] = 50,
    rng: np.random.RandomState = DEFAULT_RNG,

) -> ControlVsRespData:
    """Makes data for the control v.s. response problem.

    Args:
        model (interfere.dynamics.base.DynamicModel): The model to simulate.
        num_train_obs (int): Number of training observations.
        num_forecast_obs (int): Number of forecast time points.
        timestep (float): Timestep between each observation.
        obs_intervention (ExogIntervention): Intervention controlling 
            exogenous states during the observational stages (training and 
            forecasting).
        do_intervention (ExogIntervention): Intervention controlling   
            observational exogenous states (obs_intervention) AND intervened 
            states.
        train_prior_states (Optional[np.ndarray]): The initial state and
            historical states to use to start off simualation.
        lags (Optional[int] = 50): How many historic states there should be.
            Used only when train_prior_states is none.
        rng (np.random.Generator): Random state.

    Returns:
        A ControlVsRespData dataclass. Training data 
    """
    if train_prior_states is None:
        train_prior_states = rng.random((lags, model.dim))
    else:
        if (len(train_prior_states.shape) != 2) | \
            (train_prior_states.shape[1] != model.dim):
            raise ValueError(
                "The train_prior_states argument must be two dimensional and "
                "number of columns must match the dimension of the passed "
                "model. "
                f"\ntrain_prior_states.shape = {train_prior_states.shape}"
                f"\nmodel.dim = {model.dim}"
            )
        
        # Adjust lags.
        lags = train_prior_states.shape[0]


    train_prior_t = np.arange((-lags + 1) * timestep, 1 * timestep, timestep)
    train_t = np.arange(0, num_train_obs * timestep, timestep)

    all_times = np.hstack([train_prior_t[:-1], train_t])
    try:
        obs_intervention.eval_at_times(all_times)
    except Exception as e:
        raise ValueError(
            "Observational intervention not defined for all time points. "
            "(Sometimes caused by DataGenerator num_burn_in_states.)"
            f"\nError: \n{e} \n\n{traceback.format_exc()}"
        )
    try:
        do_intervention.eval_at_times(all_times)
    except Exception as e:
        raise ValueError(
            "\"Do intervention\" not defined for all time points. "
            "(Sometimes caused by DataGenerator num_burn_in_states.)"
            f"\nError: \n{e} \n\n{traceback.format_exc()}"
        )

    # Simulate training data.
    train_states = model.simulate(
        train_t, 
        prior_states=train_prior_states,
        prior_t=train_prior_t,
        intervention=obs_intervention,
        rng=rng
    )

    forecast_t = np.arange(
        train_t[-1],
        train_t[-1] + num_forecast_obs * timestep, 
        timestep
    )

    # Simulate forecast.
    forecast_states = model.simulate(
        forecast_t,
        prior_states=train_states,
        prior_t=train_t,
        intervention=obs_intervention,
        rng=rng
    )

    # Simulate intervention.
    interv_states = model.simulate(
        forecast_t, prior_states=train_states, prior_t=train_t, 
        intervention=do_intervention, rng=rng
    )

    return ControlVsRespData(
        train_prior_t,
        train_prior_states,
        obs_intervention,
        train_t,
        train_states,
        forecast_t,
        forecast_states,
        do_intervention,
        interv_states
    )


def make_predictions(
    method: interfere.ForecastMethod,
    data: ControlVsRespData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uses passed method to forecast training, control, and response data.

    Args:
        method (interfere.ForecastMethod): A forecasting method.
        data (ControlVsRespData): Data class containing control vs response c
            causal prediction data.

    Returns:
        train_pred (np.ndarray): Attempt to recreate training data.
        forecast_pred (np.ndarray): Attempt to forecast timesteps `forecast_t`
            that occur directly after the training data.
        interv_pred (np.ndarray): Attempt to forecast timesteps `forecast_t`
            that occur directly after the training data in response to the `do_intervention`.
    """
    # Unpack data.
    train_prior_t = data.train_prior_t
    train_prior_states = data.train_prior_states
    train_t = data.train_t
    train_states = data.train_states
    forecast_t = data.forecast_t
    obs_intervention = data.obs_intervention
    do_intervention = data.do_intervention

    # Forecast prediction.
    method.fit(train_t, *obs_intervention.split_exog(train_states))
    forecast_pred = method.simulate(
        forecast_t,
        prior_states=train_states,
        prior_t=train_t,
        intervention=obs_intervention
    )

    # Recreate training data.
    train_pred = method.simulate(
        train_t,
        prior_states=train_prior_states,
        prior_t=train_prior_t,
        intervention=obs_intervention,
    )

    # Split up endog and exog for intervention forecasting.
    method.fit(train_t, *do_intervention.split_exog(train_states))
    interv_pred = method.simulate(
        forecast_t,
        prior_states=train_states,
        prior_t=train_t,
        intervention=do_intervention
    )

    return train_pred, forecast_pred, interv_pred


def visualize(
    model: interfere.dynamics.base.DynamicModel,
    method_type: Type[interfere.ForecastMethod],
    data: ControlVsRespData,
    train_pred: np.ndarray,
    forecast_pred: np.ndarray,
    interv_pred: np.ndarray,
) -> PIL.Image:
    """Visualizes the control vs response forecasting problem.

    Args:
        model (interfere.dynamics.base.DynamicModel): A model from interfere.
        method_type (Type[interfere.ForecastMethod]): A
            forecasting method from interfere.
        data (ControlVsRespData): A data class containing the training data and
            forecasting targets.
        train_pred (np.ndarray): Attempt to recreate training data.
        forecast_pred (np.ndarray): Attempt to forecast timesteps `forecast_t`
            that occur directly after the training data.
        interv_pred (np.ndarray): Attempt to forecast timesteps `forecast_t`
            that occur directly after the training data in response to the
            `do_intervention`.
            
    Returns:
        img (PIL.Image): A visualization of the control vs response forecasting 
            problem.
    """

    dim = data.train_states.shape[1]
    fig, ax = plt.subplots(dim, 4, figsize=(12, 2 * dim))

    # For each variable in the data:
    for i in range(dim):

        # Training states.
        ax[i, 0].plot(
            data.train_t, data.train_states[:, i], c="gray", 
            label=f"Train Data"
        )
        ax[i, 0].plot(
            data.train_t, train_pred[:, i], c="r", 
            label=f"Train Data \nReproduction"
        )

        # Forecasted states.
        ax[i, 1].plot(
           data.forecast_t, data.forecast_states[:, i], c="gray",
            label=f"Forecast Truth"
        )
        ax[i, 1].plot(
            data.forecast_t, forecast_pred[:, i], c="pink",
            label=f"Forecast Pred"
        )

        # Counterfactual states.
        ax[i, 2].plot(
            data.forecast_t, data.interv_states[:, i], c="green",
            label=f"Intervention Truth"
        )
        ax[i, 2].plot(
            data.forecast_t, interv_pred[:, i], c="red", 
            label=f"Intervention Pred"
        )

        # Control-treatment plot.
        ax[i, 3].plot(
            data.forecast_t, data.forecast_states[:, i], c="gray", 
            label=f"Control Truth"
        )
        ax[i, 3].plot(
            data.forecast_t, forecast_pred[:, i], c="pink",
            label=f"Control Pred"
        )
        ax[i, 3].plot(
            data.forecast_t, data.interv_states[:, i], c="green", 
            label=f"Intervention Truth"
        )
        ax[i, 3].plot(
            data.forecast_t, interv_pred[:, i], c="red", 
            label=f"Intervention Prediction"
        )

    # Labels:
    ax[0, 0].set_title("Training Data")
    ax[0, 1].set_title("Forecast")
    ax[0, 2].set_title("Intervention")
    ax[0, 3].set_title("Control-Treatment")

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[0, 2].legend()
    ax[0, 3].legend(loc=(1.1, 0.2))

    for i in range(dim):
        ax[i, 0].set_ylabel(f"state{i}")

    for i in range(4):
        ax[-1, i].set_xlabel("time")

    # Set main title.
    plt.suptitle(
        f"Counterfactual Predictions: {method_type.__name__} on"
        f" {type(model).__name__} \n"
    )
    plt.tight_layout()
    fig.canvas.draw()

    tmp_file = TemporaryFile()
    plt.savefig(tmp_file)

    # Convert to PIL.Image.
    img = PIL.Image.open(tmp_file)
    plt.close(fig)
    
    return img


def show_pil(img: PIL.Image.Image):
    """Displays a PIL image with matplotlib.
    
    Args:
        img: A PIL image.
    """
    _, ax = plt.subplots(figsize=tuple(int(s/100) for s in img.size))
    ax.imshow(img)
    plt.axis('off')
    plt.show()


class CVROptunaObjective:
    """A class that implements an optuna objective function for control vs
    response forecasting problems.
    """

    # Replace forecasted nans with this value.
    repl_nan_val = 1e9

    def __init__(
        self,
        model: interfere.dynamics.base.DynamicModel,
        method_type: Type[interfere.ForecastMethod],
        num_train_obs: int = 100,
        num_forecast_obs: int = 25,
        timestep: float = 1.0,
        obs_intervention: Optional[ExogIntervention] = IdentityIntervention(),
        do_intervention: Optional[ExogIntervention] = IdentityIntervention(),
        train_prior_states: Optional[np.ndarray] = None,
        lags: Optional[int] = 50,
        hyperparam_func: Optional[
            Callable[[Trial], Dict[str, Any]]] = None,
        metrics: Iterable[interfere.metrics.CounterfactualForecastingMetric] = (
            interfere.metrics.ValidPredictionTime(),
            interfere.metrics.RootMeanStandardizedSquaredError(),
            interfere.metrics.TTestDirectionalChangeAccuracy()
        ),
        metric_directions: Iterable[str] = ("maximize", "minimize", "maximize"),
        store_plots: bool = True,
        store_preds: bool = True,
        raise_errors: bool = False,
        rng: np.random.RandomState = DEFAULT_RNG,
    ):
        """Initializes objective function for optuna parameter tuning.
        
        Args:
            model (interfere.dynamics.base.DynamicModel): A model from
                interfere. 
            method_type (Type[interfere.ForecastMethod]): A
                forecasting method from interfere.
            num_train_obs (int): Number of training observations.
            num_forecast_obs (int): Number of forecast time points.
            timestep (float): Timestep between each observation.
            obs_intervention (ExogIntervention): Intervention controlling exogenous   
                states during the observational stages (training and 
                forecasting).
            do_intervention (ExogIntervention): Intervention controlling   
                observational exogenous states (obs_intervention) AND 
                intervened states.
            train_prior_states (Optional[np.ndarray] = None): The initial state and
                historical states to use to start off simualation.
            lags (Optional[int] = 50): How many historic states there should be.
                Used only when train_prior_states is none.
            hyperparam_func (callable): Accepts an optuna Trial object and
                returns a dictionary of parameters. Defaults to the hyper
                parameter function built into interfere methods.
            metrics (Iterable[CounterfactualForecastingMetric]): A collection of
                metrics for measuring success at the counterfactual forecasting
                problem.
            metric_directions (Iterable[str]): Must only contain "maximize" or
                "minimize". To be passed to the optuna study. 
            store_plots (bool): Denotes if a plot of each control v.s. response 
                prediction should be saved. Accessible in self.trial_imgs.
            store_preds (bool). Accessible in self.trial_preds.
            raise_errors (bool): Toggle whether to raise errors or log them.
            rng (np.random.RandomState): Random state for reproducibility.
        """

        self.model = model
        self.method_type = method_type
        self.data = generate_data(
            model=model,
            num_train_obs=num_train_obs,
            num_forecast_obs=num_forecast_obs,
            timestep=timestep,
            obs_intervention=obs_intervention,
            do_intervention=do_intervention,
            train_prior_states=train_prior_states,
            lags=lags,
            rng=rng
        )
        self.obs_intervention = obs_intervention
        self.do_intervention = do_intervention

        if hyperparam_func is None:
            hyperparam_func = method_type._get_optuna_params 
        self.hyperparam_func = hyperparam_func
        self.metrics = metrics
        self.store_plots = store_plots
        self.store_preds = store_preds
        self.raise_errors = raise_errors
        self.metric_directions = metric_directions * 3
        self.metric_names = [
            series + m.name for m in self.metrics for series in ["train_", "forecast_", "interv_"]
        ]
        self.trial_error_log = {}
        self.trial_imgs = {}
        self.trial_preds = {}


    def compute_metrics(
        self,
        data: ControlVsRespData,
        train_pred: np.ndarray,
        forecast_pred: np.ndarray,
        interv_pred: np.ndarray,
        obs_intervention: ExogIntervention,
        do_intervention: ExogIntervention,
    ):
        """Computes metrics for control vs response problem.

        Args:
            data (ControlVsRespData): A data class containing the training data 
                and forecasting targets.
            train_pred (np.ndarray): Attempt to recreate training data.
            forecast_pred (np.ndarray): Attempt to forecast timesteps
                `data.forecast_t` that occur directly after the training data.
            interv_pred (np.ndarray): Attempt to forecast timesteps 
                `data.forecast_t` that occur directly after the training data in response to the `do_intervention`.
            obs_intervention (ExogIntervention): Intervention controlling 
                exogenous states during the observational stages (training and 
                forecasting).
            do_intervention (ExogIntervention): Intervention controlling   
                observational exogenous states (obs_intervention) AND 
                intervened  states.

        Returns:
            scores (List[float]): A list of scalar scores.
        """
        obs_exog_idxs = obs_intervention.intervened_idxs
        train_scores = [
            m(data.train_states, data.train_states, train_pred, obs_exog_idxs)
            for m in self.metrics
        ]

        forecast_scores = [
            m(
                data.train_states,
                data.forecast_states,
                forecast_pred, 
                obs_exog_idxs
            )
            for m in self.metrics
        ]

        do_exog_idxs = do_intervention.intervened_idxs
        interv_scores = [
            m(data.train_states, data.interv_states, interv_pred, do_exog_idxs)
            for m in self.metrics
        ]

        return train_scores + forecast_scores + interv_scores


    def __call__(self, trial):
        """Objective function for hyperparameter tuning."""

        # Initialize method.
        method = self.method_type(**self.hyperparam_func(trial))

        try:
            # Make predictions.
            train_pred, forecast_pred, interv_pred = make_predictions(
                method=method,
                data=self.data
            )

            # Replace nans.
            for pred in [train_pred, forecast_pred, interv_pred]:
                pred[np.isnan(pred)] = self.repl_nan_val

            # Compute scores.
            metrics = self.compute_metrics(
                data=self.data,
                train_pred=train_pred,
                forecast_pred=forecast_pred,
                interv_pred=interv_pred,
                obs_intervention=self.obs_intervention,
                do_intervention=self.do_intervention,
            )

        except Exception as e:

            if self.raise_errors:
                raise e
            
            # Store error log
            error_log = str(e) + "\n\n" + traceback.format_exc()
            self.trial_error_log[trial.number] = error_log

            # Optionally store empty predictions.
            if self.store_preds:
                self.trial_preds[trial.number] = None

            # Optionally store empty plot.
            self.trial_imgs[trial.number] = None

            # Return NANs.
            return [np.nan] * 3 * len(self.metrics)

        # Store clean error log.
        self.trial_error_log[trial.number] = ""

        # Optionally store predictions.
        if self.store_preds:
            self.trial_preds[trial.number] = {
                "train_pred": train_pred,
                "forecast_pred": forecast_pred,
                "interv_pred": interv_pred,
            }

        # Optionally store a plot.
        if self.store_plots:
            self.trial_imgs[trial.number] = visualize(
                self.model,
                self.method_type,
                self.data,
                train_pred,
                forecast_pred,
                interv_pred,
            )

        return metrics