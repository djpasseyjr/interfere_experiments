from dataclasses import dataclass
from typing import Optional, Tuple, Type

import interfere
from interfere.interventions import ExogIntervention
import matplotlib.pyplot as plt
import numpy as np
import PIL


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
        intervention (ExogIntervention): The intervention that was applied.
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
    train_t: np.ndarray
    train_states: np.ndarray
    forecast_t: np.ndarray
    forecast_states: np.ndarray
    intervention: interfere.interventions.ExogIntervention
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
        

def generate_data(
    model: interfere.dynamics.base.DynamicModel,
    num_train_obs: int,
    num_forecast_obs: int,
    timestep: float,
    intervention: interfere.interventions.ExogIntervention,
    rng: np.random.Generator,
    train_prior_states: Optional[np.ndarray] = None,
    lags: Optional[int] = 50,
) -> ControlVsRespData:
    """Makes data for the control v.s. response problem.

    Args:
        model (interfere.dynamics.base.DynamicModel): The model to simulate.
        num_train_obs (int): Number of training observations.
        num_forecast_obs (int): Number of forecast time points.
        timestep (float): Timestep between each observation.
        intervention (interfere.interventions.ExogIntervention) The intervention
            to use to generate a response.
        train_prior_states (Optional[np.ndarray] = None): The initial state and
            historical states to use to start off simualation.
        lags (Optional[int] = 50): How many historic states there should be.
            Used only when train_prior_states is none.
        rng (np.random.Generator): Random state.

    Returns:
        A ControlVsRespData dataclass. Training data 
    """
    if train_prior_states is None:
        train_prior_states = rng.random((lags, model.dim))


    train_prior_t = np.arange((-lags + 1) * timestep, 1 * timestep, timestep)
    train_t = np.arange(0, num_train_obs * timestep, timestep)

    # Simulate training data.
    train_states = model.simulate(
        train_t, prior_states=train_prior_states, prior_t=train_prior_t)

    forecast_t = np.arange(
        train_t[-1],
        train_t[-1] + num_forecast_obs * timestep, 
        timestep
    )

    # Simulate forecast.
    forecast_states = model.simulate(
        forecast_t, prior_states=train_states, prior_t=train_t)

    # Simulate intervention.
    interv_states = model.simulate(
        forecast_t, prior_states=train_states, prior_t=train_t, intervention=intervention)

    return ControlVsRespData(
        train_prior_t,
        train_prior_states,
        train_t,
        train_states,
        forecast_t,
        forecast_states,
        intervention,
        interv_states
    )


def make_predictions(
    method: interfere.methods.base.BaseInferenceMethod,
    train_prior_t: np.ndarray,
    train_prior_states: np.ndarray,
    train_t: np.ndarray,
    train_states: np.ndarray,
    forecast_t: np.ndarray,
    intervention: interfere.interventions.ExogIntervention,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uses passed method to forecast training, control, and response data.

    Args:
        method (interfere.methods.base.BaseInferenceMethod): A forecasting
            method.
        train_prior_t (ndarray): Time values corresponding to
            `train_prior_states`.
        train_prior_states (np.ndarray): Historic states relative to
            `train_states`.
        train_t (np.ndarray): Time values corresponding to `train_states`.
        train_states (np.ndarray): Training time series for the method.
        forecast_t (np.ndarray): Time values to forecast.
        intervention (interfere.interventions.ExogIntervention): Intervention
            to simulate.

    Returns:
        train_pred (np.ndarray): Attempt to recreate training data.
        forecast_pred (np.ndarray): Attempt to forecast timesteps `forecast_t`
            that occur directly after the training data.
        interv_pred (np.ndarray): Attempt to forecast timesteps `forecast_t`
            that occur directly after the training data in response to the `intervention`.
    """
    # Forecast prediction.
    method.fit(train_t, train_states)
    forecast_pred = method.predict(
        forecast_t, prior_endog_states=train_states, prior_t=train_t)

    # Recreate training data.
    train_pred = method.predict(
        train_t, prior_endog_states=train_prior_states, prior_t=train_prior_t)

    # Split up endog and exog for intervention forecasting.
    endog_tr_states, exog_tr_states = intervention.split_exog(
        train_states)
    method.fit(
        train_t, endog_states=endog_tr_states, exog_states=exog_tr_states)

    interv_exog = intervention.eval_at_times(forecast_t)

    # Forecast the intervention response.
    interv_endog_pred = method.predict(
        forecast_t,
        prior_endog_states=endog_tr_states,
        prior_t = train_t,
        prior_exog_states=exog_tr_states,
        prediction_exog=interv_exog,
    )

    # Recombine endog prediction with exogenous data.
    interv_pred = intervention.combine_exog(
        interv_endog_pred, interv_exog)

    return train_pred, forecast_pred, interv_pred


def visualize(
    model: interfere.dynamics.base.DynamicModel,
    method_type: Type[interfere.methods.base.BaseInferenceMethod],
    data: ControlVsRespData,
    train_pred: np.ndarray,
    forecast_pred: np.ndarray,
    interv_pred: np.ndarray,
) -> PIL.Image:
    """Visualizes the control vs response forecasting problem.

    Args:
        model (interfere.dynamics.base.DynamicModel): A model from interfere.
        method_type (Type[interfere.methods.base.BaseInferenceMethod]): A
            forecasting method from interfere.
        data (ControlVsRespData): A data class containing the training data and
            forecasting targets.
        train_pred (np.ndarray): Attempt to recreate training data.
        forecast_pred (np.ndarray): Attempt to forecast timesteps `forecast_t`
            that occur directly after the training data.
        interv_pred (np.ndarray): Attempt to forecast timesteps `forecast_t`
            that occur directly after the training data in response to the
            `intervention`.
            
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

    # Convert to PIL.Image.
    img = PIL.Image.frombytes(
        'RGBa', fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
    
    return img