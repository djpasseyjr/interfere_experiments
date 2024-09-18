from dataclasses import dataclass
from typing import Optional

import interfere
from interfere.interventions import ExogIntervention
import numpy as np


@dataclass
class ControlVsResponse:
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
    intervention: ExogIntervention
    interv_states: np.ndarray


def make_control_vs_response_data(
    model: interfere.dynamics.base.DynamicModel,
    num_train_obs: int,
    num_forecast_obs: int,
    timestep: float,
    intervention: interfere.interventions.ExogIntervention,
    rng: np.random.Generator,
    train_prior_states: Optional[np.ndarray] = None,
    lags: Optional[int] = 50,
) -> ControlVsResponse:
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
        A ControlVsResponse dataclass. Training data 
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

    return ControlVsResponse(
        train_prior_t,
        train_prior_states,
        train_t,
        train_states,
        forecast_t,
        forecast_states,
        intervention,
        interv_states
    )