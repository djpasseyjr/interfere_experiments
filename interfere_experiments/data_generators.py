"""Contains a collection of pre-built models from interfere.

Designed to simplify control vs response data generation by handling initial condition, stepsize and intervention.
"""
from typing import Any, Dict, Optional, Type

import interfere
from interfere.interventions import ExogIntervention
import numpy as np
import scipy.interpolate

import interfere_experiments.control_vs_resp as control_vs_resp


SEED = 11
RNG = np.random.default_rng()

class DataGenerator:


    def __init__(
        self,
        model_type: Type[interfere.DynamicModel],
        model_params: Dict[str, Any] = {},
        obs_intervention_type: Type[ExogIntervention] = interfere.interventions.IdentityIntervention,
        obs_intervention_params: Dict[str, Any] = {},
        do_intervention_type: Type[ExogIntervention] = interfere.PerfectIntervention,
        do_intervention_params: Dict[str, Any]={},
        initial_condition: np.ndarray = np.ndarray([0]),
        start_time: float = 0,
        timestep: float = 1,
        rng = np.random.default_rng(SEED)
    ):
        """Initializes an DataGenerator for generating data.

        Args:
            model_type (Type[interfere.DynamicModel]): The type of the model to
            simulate.
            model_params (Dict[str, Any]): Parameters for the model.
            obs_intervention_type (Type[ExogIntervention]):
                The type of intervention controling observational 
                exogenous states.
            obs_intervention_params (Dict[str, Any]): Parameters for the   
                observational intervention.
            do_intervention_type (Type[ExogIntervention]):
                The type of intervention controling observational 
                exogenous states and causal intervention states.
            do_intervention_params (Dict[str, Any]): Parameters for the   
                intervention.
            initial_conds (np.ndarray): Initial conditions for the model.
            start_time (float): Start time for the simulation.
            timestep (float): Timestep between each observation.
            rng (np.random.Generator): Random state.
        """
        self.model_type = model_type
        self.model_params = model_params
        self.obs_intervention_type = obs_intervention_type
        self.obs_intervention_params = obs_intervention_params
        self.do_intervention_type = do_intervention_type
        self.do_intervention_params = do_intervention_params
        self.initial_condition = initial_condition
        self.start_time = start_time
        self.timestep = timestep
        self.rng = rng


    def generate_data(
        self,
        num_train_obs: int = 100,
        num_forecast_obs: int = 50,
        num_burn_in_states: int = 0,
    ) -> control_vs_resp.ControlVsRespData:
        """Generates data from the model.

        Args:
            num_train_obs (int): The number of traing data observations to
                generate. Defaults to 100.
            num_forecast_obs (int): The number of forecast data observations to
                generate. Defaults to 50.
            num_burn_in_obs (int): Number of observations to chop off the beginning 
                of the generated data. This is used to remove the effect of 
                transient states due to the initial condition.

        Returns:
            ControlVsRespData: The generated data.
        """
        if (
            (num_burn_in_states < 0) or (num_forecast_obs < 0) or (num_train_obs < 0)
        ):
            raise ValueError(
                f"The variables num_train_obs = {num_train_obs}, "
                "num_forecast_obs = {num_forecast_obs}, and num_burn_in_states "
                "= {num_burn_in_states} must be greater than zero."
            )
        
        data = control_vs_resp.generate_data(
            self.model_type(**self.model_params),
            num_train_obs=num_train_obs + num_burn_in_states,
            num_forecast_obs=num_forecast_obs,
            timestep=self.timestep,
            do_intervention=self.do_intervention_type(
                **self.do_intervention_params),
            obs_intervention=self.obs_intervention_type(
                **self.obs_intervention_params
            ),
            train_prior_states=self.initial_condition,
            rng=self.rng
        )

        # Collect all training data
        all_train_states = np.vstack([
            # Drop last prior state because it is same as the first train state.
            data.train_prior_states[:-1, :],
            data.train_states
        ])

        # Drop last prior time because it is same as the first train time.
        all_times = np.hstack([data.train_prior_t[:-1], data.train_t])

        num_prior_obs = len(data.train_prior_t)

        # Remove burn in state observations.
        data.train_states = all_train_states[-num_train_obs:]
        data.train_prior_states = all_train_states[
            -(num_prior_obs + num_train_obs) + 1:(-num_train_obs + 1)
        ]

        # Adjust burn in times to match states.
        data.train_t = all_times[-num_train_obs:]
        data.train_prior_t = all_times[
            -(num_prior_obs + num_train_obs) + 1:(-num_train_obs + 1)
        ]
        return data


    def generate_data_and_downsample(
        self, 
        new_timestep,
        num_train_obs = 100,
        num_forecast_obs = 50,
        num_burn_in_states = 0,
    ) -> control_vs_resp.ControlVsRespData:
        """Generates data using new_timestep and then downsamples it.
        Used for dynamics where timestep must be small to prevent numerical
        instabilities.

        Args:
            new_timestep (float): The timestep to downsample to.
            num_train_obs (int): Number of training observations.
            num_forecast_obs (int): Number of forecast observations.
            num_burn_in_states (int): Number of burn-in states.

        Returns:
            Data (ControlVsRespData): The generated data.

        """
        if isinstance(
            self.model_type, interfere.dynamics.base.DiscreteTimeDynamics):
            raise NotImplementedError(
                "Down sampling not implemented for discrete time dynamics"
            )

        if self.timestep < new_timestep:
            return self.generate_data(
                num_train_obs,
                num_forecast_obs,
                num_burn_in_states
            )

        else:
            # Simulate with smaller timestep and downsample.
            down_sample_rate = int(self.timestep / new_timestep)
            old_timestep = self.timestep
            self.timestep = self.timestep / down_sample_rate

            # Adjust initial condition.
            old_initial_cond = self.initial_condition

            # If there is only a single prior state, make copies.
            if old_initial_cond.shape[0] == 1:
                new_initial_cond = np.vstack(
                    [old_initial_cond] * down_sample_rate
                )

            # Else, increase the number of prior observations via interpolation.
            else:
                prior_t = np.arange(old_initial_cond.shape[0])

                initial_cond_interp = scipy.interpolate.interp1d(
                    prior_t, old_initial_cond, axis=0)
                
                dt = 1 / down_sample_rate
                new_prior_t = np.arange(
                    0, old_initial_cond.shape[0] - 1 + dt, dt)
                new_initial_cond = initial_cond_interp(new_prior_t)

            self.initial_condition = new_initial_cond
            
            data = self.generate_data(
                num_train_obs * down_sample_rate,
                num_forecast_obs * down_sample_rate,
                num_burn_in_states * down_sample_rate
            )

            # Down sample data fields.
            data.train_prior_t = data.train_prior_t[::down_sample_rate]
            data.train_prior_states = data.train_prior_states[::down_sample_rate]
            data.train_t = data.train_t[::down_sample_rate]
            data.train_states = data.train_states[::down_sample_rate]
            data.forecast_t = data.forecast_t[::down_sample_rate]
            data.forecast_states = data.forecast_states[::down_sample_rate]
            data.interv_states = data.interv_states[::down_sample_rate]

            self.timestep = old_timestep
            self.initial_condition = old_initial_cond
            return data


def uniform_frequency_dist_random_signal(
    total_time: float,
    amplitude_min: float = 0.0,
    amplitude_max: float = 1.0,
    frequency_min: float = 2.0,
    frequency_max: float = 10.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Creates a signal with a uniform dist of frequencies and amplitudes.

    Args:
        total_time (float): Total time in seconds.
        amplitude_min (float): Minimum amplitude.
        amplitude_max (float): Maximum amplitude.
        frequency_min (float): Minimum frequency in oscillations per second.
            Must be greater than zero.
        frequency_max (float): Maximum frequency in oscillations per second.
            Must be greater than frequency_min.
        rng (np.random.RandomState): Random number generator.

    Returns:
        times (np.ndarray): Time values for a signal with a uniform dist of
            frequencies and amplitudes.
        values (np.ndarray): Values for a signal with a uniform dist of
            frequencies and amplitudes.
    """
    if total_time <= 0:
        raise ValueError("total_time must be greater than 0")

    if 1/frequency_min >= total_time:
        raise ValueError("1/frequency_min must be less than total_time")

    if frequency_max < frequency_min:
        raise ValueError("frequency_max must be greater than frequency_min")

    if amplitude_max <= amplitude_min:
        raise ValueError("amplitude_max must be greater than amplitude_min")

    if rng is None:
        rng = np.random.default_rng()

    rand_val = lambda : rng.random() * (
        amplitude_max - amplitude_min) + amplitude_min
    rand_freq = lambda : rng.random() * (
        frequency_max - frequency_min) + frequency_min

    times = [0]
    values = [rand_val()]
    while times[-1] < total_time:
        times.append(times[-1] + 1/rand_freq())
        values.append(rand_val())

    return np.array(times), np.array(values)


def randsig(max_T, amin=-1, amax=1, fmin=2, fmax=10, rng=RNG):
    """Creates function that interpolates a uniform freq random signal.
    
    Args:
        max_T (float): Domain of the function will be [0, max_T].
        amin (float): Minimum amplitude.
        amax (float): Maximum amplitude.
        fmin (float): Minimum frequency in oscillations per second.
            Must be greater than zero.
        fmax (float): Maximum frequency in oscillations per second.
            Must be greater than frequency_min.
        rng (np.random.RandomState): Random number generator.

    Returns:
        CubicSpline: A cubic spline interpolation of a randomly oscilating 
            signal.
    """
    return scipy.interpolate.CubicSpline(
        *uniform_frequency_dist_random_signal(
            max_T, amplitude_min=amin, amplitude_max=amax,
            frequency_min=fmin, frequency_max=fmax, rng=rng
        ),
        bc_type="clamped", extrapolate=False,
    )


class ArithmeticBrownianMotion(DataGenerator):


    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.ArithmeticBrownianMotion,
            model_params={
                "mu": np.array([0.1, -0.3, 0.06, -0.01, -0.2,
                                -0.05, 0.15, 0.22, -0.17, -0.01]), 
                "sigma": np.zeros(10),
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 4.0},
            initial_condition=np.array([
                [0.88706501, -0.28115793,  0.56961082,  0.18255637,
                -0.41134288, 0.84545137,  0.73866309, -0.27172315,  
                0.94635363, -0.55095134],
                [ 0.8388921 ,  0.0247688 ,  0.10156725, -0.29358369, 
                 -0.94296698, -0.2503428 ,  0.97018181,  0.01633963,  
                  0.61435139, -0.67821191]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )


class AttractingFixedPoint4D(DataGenerator):

    target_idx = 3
    interv_sig = lambda self, t: 0.4
    exog = randsig(
        max_T=11_000, amax=0.6, amin=0.4,
        fmax=3, fmin=0.2, rng=np.random.default_rng(11)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.attracting_fixed_point_4d_linear_sde,
            model_params={"sigma": 0.0},
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2],
                "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2],
                "signals": [self.interv_sig, self.exog]
            },
            
            initial_condition=np.array([
                [-0.73907002, -0.65245823, -0.48475903,  0.85564824],
                [ 0.61719957, -0.9664203 ,  0.12427135, -0.59981189]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )


class Belozyorov1(DataGenerator):

    target_idx = 0

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Belozyorov3DQuad,
            model_params={
                "mu": 1.81,
                "sigma": 0.0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 1, "constants": 1.0},
            initial_condition=np.array([
                [0.66158597, 0.8012904 , 0.19920669],
                [0.8102566 ,  0.10224813, -0.35046573]
            ]),
            start_time=0,
            timestep=0.02,
            rng = np.random.default_rng(SEED)
        )


class Belozyorov2(DataGenerator):
        
    target_idx = 0

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Belozyorov3DQuad,
            model_params={
                "mu": 1.4,
                "sigma": 0.0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 2, "constants": -1.0},
            initial_condition=np.array([
                [0.66158597, 0.8012904 , 0.19920669],
                [0.8102566 ,  0.10224813, -0.35046573]
            ]),
            start_time=0,
            timestep=0.02,
            rng = np.random.default_rng(SEED)
        )


class Belozyorov3(DataGenerator):

    target_idx = 0

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Belozyorov3DQuad,
            model_params={
                "mu": 2.2,
                "sigma": 0.0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 1, "constants": 1.0},
            initial_condition=np.array([
                [0.66158597, 0.8012904 , 0.19920669],
                [0.8102566 ,  0.10224813, -0.35046573]
            ]),
            start_time=0,
            timestep=0.02,
            rng = np.random.default_rng(SEED)
        )

class CoupledLogisticMapAllToAll(DataGenerator):

    target_idx = 1
    interv_sig = lambda self, t: 0.9
    exog = randsig(
        max_T=11000, amax=0.6, amin=0.4,
        fmax=3, rng=np.random.default_rng(11)
    )


    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.coupled_logistic_map,
            model_params={
                "adjacency_matrix": np.ones((6, 6)),
                "eps": 0.8,
                "logistic_param": 3.72,
                "sigma": 0.0,
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2],
                "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2],
                "signals": [self.interv_sig, self.exog]
            },
            initial_condition=np.array(
                [[0.26974736, 0.36818973, 0.30747773, 0.69532405, 0.07728317, 
                 0.38090953]]
            ),
            start_time=0,
            timestep=1,
            rng = np.random.default_rng(SEED)
        )


# Coupled Map Lattice Data Generators
class CoupledLogisticMapConfound(DataGenerator):

    target_idx = 1

    # Intervention signals
    interv_sig = lambda self, t: 0.4
    exog1 = randsig(
        max_T=11_000, amax=0.6, amin=0.4,
        fmax=3, fmin=0.2, rng=np.random.default_rng(11) 
    )
    exog2 = randsig(
        max_T=11_000, amax=0.6, amin=0.4,
        fmax=3, fmin=0.2, rng=np.random.default_rng(14)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.coupled_logistic_map,
            model_params={
                "logistic_param": 3.72,
                "eps": 0.5,
                "adjacency_matrix": 0.5 * np.array([
                    [1.0, 0.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0],
                ]),
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2, 3],
                "signals": [self.exog1, self.exog2]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2, 3],
                "signals": [self.interv_sig, self.exog1, self.exog2]
            },
            initial_condition=np.array([[0.69274337, 0.81581711, 0.34440676, 0.04483818, 0.57159726]]),
            timestep=1.0,
        )


class CoupledLogisticMapBlockableConfound(DataGenerator):

    target_idx = 1


    # Intervention signals
    interv_sig = lambda self, t: 0.4
    exog1 = randsig(
        max_T=11_000, amax=0.6, amin=0.4,
        fmax=3, fmin=0.2, rng=np.random.default_rng(11)
    )
    exog2 = randsig(
        max_T=11_000, amax=0.6, amin=0.4,
        fmax=3, fmin=0.2, rng=np.random.default_rng(14)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.coupled_logistic_map,
            model_params={
                "logistic_param": 3.72,
                "eps": 0.5,
                "adjacency_matrix": 0.5 *  np.array([
                    [1.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0],
                ]),
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2],
                "signals": [self.exog1]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2],
                "signals": [self.interv_sig, self.exog1]
            },
            initial_condition=np.array([[0.31268323, 0.22431822, 0.18200973, 0.86017865, 0.85754249]]),
            timestep=1.0,
        )


class CoupledLogisticMapTwoCycles(DataGenerator):

    target_idx = 3
    
    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.coupled_logistic_map,
            model_params={
                "adjacency_matrix": np.array([
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                ]),
                "eps": 0.9,
                "logistic_param": 3.72,
                "sigma": 0.0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 1.0},
            initial_condition=np.array([
                [0.44014438, 0.3384513 , 0.03194094, 0.65459332, 0.09945458],
                [0.26974736, 0.36818973, 0.30747773, 0.69532405, 0.07728317]
            ]),
            start_time=0,
            timestep=1,
            rng = np.random.default_rng(SEED)
        )


class CoupledLogisticMapOneCycle(DataGenerator):

    target_idx = 2
    
    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.coupled_logistic_map,
            model_params={
                "adjacency_matrix": 0.5 * np.array([
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0],
                ]),
                "eps": 0.3,
                "logistic_param": 3.0,
                "sigma": 0.0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 0.1},
            initial_condition=np.array([
                [0.44014438, 0.3384513 , 0.03194094, 0.65459332],
                [0.26974736, 0.36818973, 0.30747773, 0.69532405]
            ]),
            start_time=0,
            timestep=1,
            rng = np.random.default_rng(SEED)
        )


class CoupledMapLatticeChaoticBrownian(DataGenerator):

    target_idx = 6

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.coupled_map_1dlattice_chaotic_brownian,
            model_params={
                "dim": 7,
                "sigma": 0.0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 5, "constants": 0.75},
            initial_condition=np.array([
                [0.02267611, 0.90949372, 0.92080279, 0.97851563, 0.47947842,
                0.59329169, 0.2889972],
                [0.29505913, 0.35622953, 0.67069543, 0.76956994, 0.10373029,
                0.92888572, 0.75312997]
            ]),
            start_time=0,
            timestep=1,
            rng = np.random.default_rng(SEED)
        )


class CoupledMapLatticeChaoticTravelingWave(DataGenerator):

    target_idx = 2

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.coupled_map_1dlattice_chaotic_traveling_wave,
            model_params={
                "dim": 3,
                "sigma": 0.0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 0.5},
            initial_condition=np.array([
                [0.02267611, 0.90949372, 0.92080279],
                [0.29505913, 0.35622953, 0.67069543]
            ]),
            start_time=0,
            timestep=1,
            rng = np.random.default_rng(SEED)
        )



class CoupledMapLatticeDefectTurbulence(DataGenerator):

    target_idx = 4

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.coupled_map_1dlattice_defect_turbulence,
            model_params={
                "dim": 5,
                "sigma": 0.0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": -0.5},
            initial_condition=np.array([
                [0.02267611, 0.90949372, 0.92080279, 0.97851563, 0.47947842],
                [0.29505913, 0.35622953, 0.67069543, 0.76956994, 0.10373029]
            ]),
            start_time=0,
            timestep=1,
            rng = np.random.default_rng(SEED)
        )


class CoupledMapLatticePatternSelection(DataGenerator):

    target_idx = 1

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.coupled_map_1dlattice_pattern_selection,
            model_params={
                "dim": 6,
                "sigma": 0.0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": -0.5},
            initial_condition=np.array([
                [0.02267611, 0.90949372, 0.92080279, 0.97851563, 0.47947842,
                0.59329169],
                [0.29505913, 0.35622953, 0.67069543, 0.76956994, 0.10373029,
                0.92888572]
            ]),
            start_time=0,
            timestep=1,
            rng = np.random.default_rng(SEED)
        )


class CoupledMapLatticeSpatioTempChaos(DataGenerator):

    target_idx = 3

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.coupled_map_1dlattice_spatiotemp_chaos,
            model_params={
                "dim": 5,
                "sigma": 0.0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": -0.5},
            initial_condition=np.array([
                [0.02267611, 0.90949372, 0.92080279, 0.97851563, 0.47947842],
                [0.29505913, 0.35622953, 0.67069543, 0.76956994, 0.10373029]
            ]),
            start_time=0,
            timestep=1,
            rng = np.random.default_rng(SEED)
        )


class CoupledMapLatticeSpatioTempIntermit(DataGenerator):
    
    target_idx = 0

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.coupled_map_1dlattice_spatiotemp_intermit1,
            model_params={
                "dim": 6,
                "sigma": 0.0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 1, "constants": 0.5},
            initial_condition=np.array([
                [0.02267611, 0.90949372, 0.92080279, 0.97851563, 0.47947842,
                0.59329169],
                [0.29505913, 0.35622953, 0.67069543, 0.76956994, 0.10373029,
                0.92888572]
            ]),
            start_time=0,
            timestep=1,
            rng = np.random.default_rng(SEED)
        )


class CoupledMapLatticeTravelingWave(DataGenerator):

    target_idx = 1
    
    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.coupled_map_1dlattice_traveling_wave,
            model_params={
                "dim": 3,
                "sigma": 0.0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 0.5},
            initial_condition=np.array([
                [0.02267611, 0.90949372, 0.92080279],
                [0.29505913, 0.35622953, 0.67069543]
            ]),
            start_time=0,
            timestep=1,
            rng = np.random.default_rng(SEED)
        )


class DampedOscillator1(DataGenerator):

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.DampedOscillator,
            model_params={"m": 1.0, "c": 2, "k": 10, "sigma": 0},
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": -0.5},
            initial_condition=np.array([
                [0.17484967, 0.56195172],
                [-0.23292231, -0.69167961]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )


class DampedOscillator2(DataGenerator):

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.DampedOscillator,
            model_params={"m": 30.0, "c": 0.5, "k": 2, "sigma": 0, },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": -0.5},
            initial_condition=np.array([
                [0.17484967, 0.56195172],
                [-0.23292231, -0.69167961]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )


class GeometricBrownianMotion1(DataGenerator):

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.GeometricBrownianMotion,
            model_params={
                "mu": np.array([
                    -0.2, 0.003, -10, -0.0007, -1.3, -6, -0.8, -7, -0.38, -20.0]),
                "sigma": np.zeros(10),
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 0.5},
            initial_condition=np.array([
                [0.21433191, -0.19033034, -0.19308829, -0.82433803, -0.1581898,
        -0.89612777,  0.80391384, -0.25030014,  0.43063596,  0.17665972],
                [-0.11971123, -0.3230974 , -0.93611812,  0.30918664, -0.80109084,
        -0.53039582,  0.91781623, -0.24832396, -0.27500764,  0.37978088]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )


class GeometricBrownianMotion2(DataGenerator):
    
    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.GeometricBrownianMotion,
            model_params={
                "mu": np.array([
                    -0.2, 0.003, -10, -0.0007, -1.3, -6, -0.8, -7, -0.38, -20.0
                ]),
                "sigma": np.zeros(10),
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 0.5},
            initial_condition=np.array([
                [0.21433191, -0.19033034, -0.19308829, -0.82433803, -0.1581898,
        -0.89612777,  0.80391384, -0.25030014,  0.43063596,  0.17665972],
                [-0.11971123, -0.3230974 , -0.93611812,  0.30918664, -0.80109084,
        -0.53039582,  0.91781623, -0.24832396, -0.27500764,  0.37978088]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )


class HodgkinHuxley1(DataGenerator):

    target_idx = 7

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.HodgkinHuxleyPyclustering,
            model_params={
                "stimulus": np.array(
                    [ -7.04509716,  -0.61807187,  15.3283015 ,   7.5973135 ,
                    26.29443739,   6.80411478,  -4.74567746,  24.51372455,
                    -18.51728507, -10.2845044 ]
                ),
                "sigma": np.zeros(10),
                "w1": 0,
                "w2": 1.0,
                "w3": 0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": -20},
            initial_condition=np.array([
                [-26.51263192, -21.59051353, -24.62611325,  -5.23379733,
                 -36.13584145, -20.95452341,   9.8631521 ,  -7.9524096 ,
                 -16.24248747, -25.5402178 ],
                [-38.86619469,   5.47468612,   6.04013967,   8.9257813 ,
                 -16.02607912, -10.33541563, -25.55013988,   1.41989336,
                 -10.8672689 , -25.16636806]
            ]),
            start_time=0,
            timestep=0.1,
            rng = np.random.default_rng(SEED)
        )

    @interfere.utils.copy_doc(DataGenerator.generate_data)
    def generate_data(
        self, num_train_obs = 100,
        num_forecast_obs = 50,
        num_burn_in_states = 0
    ):
        # Stepsize must be small or forward Euler diverges.
        if self.timestep > 0.002:
            # Simulate with smaller timestep and downsample.
            new_timestep = 0.001
            return self.generate_data_and_downsample(
                new_timestep,
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
                num_burn_in_states=num_burn_in_states,
            )
        
        else:
            return super().generate_data(
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
            )

class HodgkinHuxley2Chain(DataGenerator):

    target_idx = 6

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.HodgkinHuxleyPyclustering,
            model_params={
                "stimulus": np.array([
                    27.70407901, 35.16803566, 29.85919101, 65.14444685,
                    61.85386246, 26.86829904, 26.31757841, 31.51959922,
                    58.18741777, 24.35036104
                ]),
                "sigma": np.zeros(10),
                "w1": 0.5,
                "w2": 0,
                "w3": 0,
                "type_conn": "list_bdir"
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 70},
            initial_condition=np.array([
                [-19.22092384,  -4.36571889, -22.0525331 , -28.14107253,
                -8.67268893, -26.42471519, -20.92204683,   1.52784374,
                -18.43917783, -24.04612304],
                [-25.49562614, -10.56734573,   9.97989427, -11.64972838,
                0.85883408, -39.30753626,   9.50537316, -15.01307497,
                1.68306979,   9.1188609 ]
            ]),
            start_time=0,
            timestep=0.1,
            rng = np.random.default_rng(SEED)
        )

    @interfere.utils.copy_doc(DataGenerator.generate_data)
    def generate_data(
        self, num_train_obs = 100,
        num_forecast_obs = 50,
        num_burn_in_states = 0
    ):
        # Stepsize must be small or forward Euler diverges.
        if self.timestep > 0.002:
            # Simulate with smaller timestep and downsample.
            new_timestep = 0.001
            return self.generate_data_and_downsample(
                new_timestep,
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
                num_burn_in_states=num_burn_in_states,
            )
        
        else:
            return super().generate_data(
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
            )


class HodgkinHuxley3Grid(DataGenerator):

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.HodgkinHuxleyPyclustering,
            model_params={
                "stimulus": np.array(
                [50.427046  , 21.77379092, 68.36204063, 65.45262642, 40.63269141,
                37.05453064, 44.31010883, 67.77972547, 53.39835176, 59.38748481]
            ),
                "sigma": np.zeros(10),
                "w1": 0.5,
                "w2": 0.5,
                "w3": 0,
                "type_conn": "grid_four"
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 70},
            initial_condition=np.array([
                [ -6.82502728,   9.82700264, -34.8376886 ,   4.36644889,
                -23.44068506, -36.10761556, -25.57154861, -35.75632685,
                -1.07734235,  -6.85013624],
                [-15.68128686,   1.06898729, -23.06349775, -14.47807305,
                -11.88460932, -16.36100571, -31.18575758,  -3.63489649,
                -28.67309254, -35.98041389]
            ]),
            start_time=0,
            timestep=0.0001,
            rng = np.random.default_rng(SEED)
        )


class ImaginaryRoots4D(DataGenerator):

    target_idx = 1

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.imag_roots_4d_linear_sde,
            model_params={
                "sigma": 0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": -0.5},
            initial_condition=np.array([
                [0.02267611, 0.90949372, 0.92080279, 0.97851563],
                [0.32654883, 0.8377457 , 0.82375113, 0.23435478]
            ]),
            start_time=0,
            timestep=0.01,
            rng = np.random.default_rng(SEED)
        )


class KuramotoOscilator1(DataGenerator):

    target_idx = 8

    interv_sig = lambda self, t: 0.4
    exog = randsig(
        max_T=11_000, amax=0.3, amin=-0.3,
        fmax=2, fmin=0.2, rng=np.random.default_rng(11)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Kuramoto,
            model_params={
                "omega": np.array([
                    1.54984929, 0.48985921, 0.73281211, 0.42658923, 1.15827222,
                    0.10765091, 1.14874596, 1.31069673, 0.38975643, 0.5632573 
                ]),
                "K": 2.5,
                "adjacency_matrix": np.array([
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ]),
                "sigma": 0,
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2],
                "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2],
                "signals": [self.interv_sig, self.exog]
            },
            initial_condition=np.array([
                [0.21688613, -0.52486516,  0.62411263, -0.2703094 , -0.82369351,
            -0.69257434, -0.25823711,  0.66971335, -0.54667034,  0.56496237],
            [-0.83527427,  0.67926305,  0.42997372, -0.82346111, -0.91524358,
            -0.49706574, -0.64154586,  0.29474006, -0.05792134,  0.14611772]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )


class KuramotoOscilator2(DataGenerator):

    target_idx = 1

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Kuramoto,
            model_params={
                "omega": np.array([
                    1.54984929, 0.48985921, 0.73281211, 0.42658923, 1.15827222,
                    0.10765091, 1.14874596, 1.31069673, 0.38975643, 0.5632573 
                ]),
                "K": 2.5,
                # A cycle and isolated node
                "adjacency_matrix": np.array([
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]),
                "sigma": 0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": -0.5},
            initial_condition=np.array([
                [0.21688613, -0.52486516,  0.62411263, -0.2703094 , -0.82369351,
            -0.69257434, -0.25823711,  0.66971335, -0.54667034,  0.56496237],
            [-0.83527427,  0.67926305,  0.42997372, -0.82346111, -0.91524358,
            -0.49706574, -0.64154586,  0.29474006, -0.05792134,  0.14611772]
            ]),
            start_time=0,
            timestep=0.075,
            rng = np.random.default_rng(SEED)
        )


class KuramotoOscilator3(DataGenerator):

    target_idx = 2

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Kuramoto,
            model_params={
                "omega": np.array([
                    1.54984929, 0.48985921, 0.73281211, 0.42658923, 1.15827222,
                    0.10765091, 1.14874596]),
                "K": 2.5,
                "adjacency_matrix": np.array([
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]),
                "sigma": 0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 0.75},
            initial_condition=np.array([
                [0.21688613, -0.52486516,  0.62411263, -0.2703094 , -0.82369351,
            -0.69257434, -0.25823711],
            [-0.83527427,  0.67926305,  0.42997372, -0.82346111, -0.91524358,
            -0.49706574, -0.64154586]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )
        

class KuramotoOscilator4(DataGenerator):

    target_idx = 1

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Kuramoto,
            model_params={
                "omega": np.array([
                    1.54984929, 0.48985921, 0.73281211
                ]),
                "K": 2.5,
                "adjacency_matrix": np.array([
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]),
                "sigma": 0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": -0.5},
            initial_condition=np.array([
                [0.21688613, -0.52486516,  0.62411263],
                [-0.83527427,  0.67926305,  0.42997372]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )


class KuramotoSakaguchi1(DataGenerator):
    
    target_idx = 1

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.KuramotoSakaguchi,
            model_params={
                "omega": np.array([
                    0.54272042, 1.37668642, 0.24483047, 1.43619725, 0.14545527,
                    0.05824583
                ]),
                "K": 2.5,
                "adjacency_matrix": np.array([
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]),
                "phase_frustration": np.array([
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ]),
                "sigma": 0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": -0.5},
            initial_condition=np.array([
                [0.21688613, -0.52486516,  0.62411263, -0.2703094 , -0.82369351,
            -0.69257434],
            [-0.83527427,  0.67926305,  0.42997372, -0.82346111, -0.91524358,
            -0.49706574]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )


class KuramotoSakaguchi2(DataGenerator):

    target_idx = 1

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.KuramotoSakaguchi,
            model_params={
                "omega": np.array([
                    0.54272042, 1.37668642, 0.24483047, 1.43619725, 0.14545527,
                ]),
                "K": 7.0,
                "adjacency_matrix": np.array([
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],

                ]),
                "phase_frustration": np.array([
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                ]),
                "sigma": 0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 0.75},
            initial_condition=np.array([
                [ 0.77161016,  0.3487197 ,  0.96420325,  0.36306674, -0.33142683,],
            [-2.27364797e-01, -6.39967817e-04, -8.92835027e-01, -1.88599377e-01,
            7.39888259e-02]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )


class KuramotoSakaguchi3(DataGenerator):

    target_idx = 1

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.KuramotoSakaguchi,
            model_params={
                # All nodes have the same fundamental frequency
                "omega": np.ones(5),
                "K": 4.0,
                "adjacency_matrix": np.array([
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],

                ]),
                "phase_frustration": np.array([
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                ]),
                "sigma": 0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": -0.5},
            initial_condition=np.array([
                [ 0.77161016,  0.3487197 ,  0.96420325,  0.36306674, -0.33142683,],
                [-2.27364797e-01, -6.39967817e-04, -8.92835027e-01, -1.88599377e-01,
            7.39888259e-02]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )


class Liping3DQuadFinance1(DataGenerator):

    target_idx = 1

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Liping3DQuadFinance,
            model_params={
                "sigma": 0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": -0.1},
            initial_condition=np.array([
                [-1.77563702, -1.52105886, -0.69282225],
                [-1.45612385,  0.54978974, -1.72199022]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )




class Lorenz1(DataGenerator):

    target_idx = 1

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Lorenz,
            model_params={},
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 15.0},
            initial_condition=np.array([
                [0.66158597, 0.8012904 , 0.19920669],
                [0.8102566 ,  0.10224813, -0.35046573]
            ]),
            start_time=0,
            timestep=0.02,
            rng = np.random.default_rng(SEED)        
        )

class Lorenz2(DataGenerator):

    target_idx = 0

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Lorenz,
            model_params={"beta": 3, "rho": 32, "s": 12},
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 1, "constants": -10.0},
            initial_condition=np.array([
                [0.66158597, 0.8012904 , 0.19920669],
                [0.8102566 ,  0.10224813, -0.35046573]
            ]),
            start_time=0,
            timestep=0.02,
            rng = np.random.default_rng(SEED)        
        )


class Liping3DQuadFinance2(DataGenerator):

    target_idx = 1

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Liping3DQuadFinance,
            model_params={
                "sigma": 0,
            },
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 2, "constants": -5.0},
            initial_condition=np.array([
                [-1.77563702, -1.52105886, -0.69282225],
                [-1.45612385,  0.54978974, -1.72199022]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)
        )


class LotkaVoltera1(DataGenerator):

    target_idx = 1

    interv_sig = lambda self, t: 0.0
    exog = randsig(
            max_T=300, amax=6, amin=3,
            fmax=2, fmin=1, rng=np.random.default_rng(11)
        )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.LotkaVolteraSDE,
            model_params={
                "growth_rates": 5 * np.array(
                    [0.54244252, 0.36324576, 0.38389436, 0.07903888, 0.71901175]),
                "capacities": 20 * np.ones(5),
                # A cycle and isolated node
                "interaction_mat": np.array([
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                ]),
                "sigma": 0,
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2],
                "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2],
                "signals": [self.interv_sig, self.exog]
            },
            initial_condition=np.array([
                [6.1456928 , 3.42103611, 7.00446391, 6.35503927, 3.11792205],
                [4.95485522, 2.36929088, 4.92437877, 7.56459594, 5.45540964],
            ]),
            start_time=0,
            timestep=0.02,
            rng = np.random.default_rng(SEED)
        )


class LotkaVoltera2(DataGenerator):

    target_idx = 1

    interv_sig = lambda self, t: 1.0
    exog = randsig(
            max_T=300, amax=4, amin=3,
            fmax=3, fmin=0.5, rng=np.random.default_rng(117)
        )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.LotkaVolteraSDE,
            model_params={
                # All nodes have the same growth rate
                "growth_rates": np.ones(10),
                "capacities": 20 * np.ones(10),
                # Three cycles.
                "interaction_mat": np.array([
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ]),
                "sigma": 0,
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [8],
                "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 8],
                "signals": [self.interv_sig, self.exog]
            },
            initial_condition=np.array([
                [6.1456928 , 3.42103611, 7.00446391, 6.35503927, 3.11792205,
                2.79971274, 3.91900287, 2.95954181, 7.58954302, 7.90596947],
                [4.95485522, 2.36929088, 4.92437877, 7.56459594, 5.45540964,
                0.15525796, 0.8760932 , 3.84472739, 8.42631288, 4.27621899]
            ]),
            start_time=0,
            timestep=0.02,
            rng = np.random.default_rng(SEED)
        )


class LotkaVoltera3(DataGenerator):

    target_idx = 4

    interv_sig = lambda self, t: 12.0
    exog1 = randsig(
            max_T=300, amax=12, amin=6,
            fmax=3, fmin=0.5, rng=np.random.default_rng(117)
        )
    exog2 = randsig(
            max_T=300, amax=12, amin=6,
            fmax=3, fmin=0.5, rng=np.random.default_rng(118)
        )
    
    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.LotkaVolteraSDE,
            model_params={
                "growth_rates": np.ones(10),
                "capacities": 20 * np.ones(10),
                # All nodes have the same growth rate
                "interaction_mat": np.array([
                    [0., 1., 1., 1., 0., 0., 0., 0., 1., 0.],
                    [0., 0., 1., 0., 1., 0., 0., 1., 0., 0.],
                    [1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 1., 0., 1.],
                    [1., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
                    [0., 0., 0., 0., 1., 1., 1., 0., 1., 0.],
                    [0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
                    [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 1., 0., 0., 1., 0.]
                ]),
                "sigma": 0,
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [1, 8],
                "signals": [self.exog1, self.exog2]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [5, 1, 8],
                "signals": [self.interv_sig, self.exog1, self.exog2]
            },
            initial_condition=np.array([
                [6.1456928 , 3.42103611, 7.00446391, 6.35503927, 3.11792205,
                2.79971274, 3.91900287, 2.95954181, 7.58954302, 7.90596947],
                [4.95485522, 2.36929088, 4.92437877, 7.56459594, 5.45540964,
                0.15525796, 0.8760932 , 3.84472739, 8.42631288, 4.27621899]
            ]),
            start_time=0,
            timestep=0.02,
            rng = np.random.default_rng(SEED)
        )


class LotkaVoltera4(DataGenerator):

    target_idx = 4

    interv_sig = lambda self, t: 12.0
    exog = randsig(
            max_T=300, amax=12, amin=6,
            fmax=3, fmin=0.5, rng=np.random.default_rng(117)
        )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.LotkaVolteraSDE,
            model_params={
                "growth_rates": 2 * np.array(
                    [0.54244252, 0.36324576, 0.38389436, 0.07903888, 0.71901175]),
                "capacities": 20 * np.ones(5),
                "interaction_mat": np.array([
                    [0., 1., 1., 1., 0.],
                    [1., 0., 0., 1., 1.],
                    [1., 1., 0., 1., 0.],
                    [1., 0., 0., 0., 1.],
                    [1., 0., 0., 1., 0.]
                ]),
                "sigma": 0,
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2],
                "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2],
                "signals": [self.interv_sig, self.exog]
            },
            initial_condition=np.array([
                [6.1456928 , 3.42103611, 7.00446391, 6.35503927, 3.11792205],
                [4.95485522, 2.36929088, 4.92437877, 7.56459594, 5.45540964],
            ]),
            start_time=0,
            timestep=0.02,
            rng = np.random.default_rng(SEED)
        )



class LotkaVolteraConfound(DataGenerator):

    target_idx = 1

    # Intervention signals
    interv_sig = lambda self, t: 7.0
    exog1 = randsig(
        max_T=300, amax=6, amin=3,
        fmax=3, rng=np.random.default_rng(11)
    )
    exog2 = randsig(
        max_T=300, amax=6, amin=3,
        fmax=3, rng=np.random.default_rng(14)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.LotkaVolteraSDE,
            model_params={
                "growth_rates": np.array([ 8.89666155,  9.27497536,  9.21819474, 10.26697586,  9.75141927]),
                "capacities": np.array([11.35919604, 16.8903648 , 18.41747724, 14.25508997, 19.56926003]),
                "interaction_mat": 0.5 * np.array([
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                ]),
                "sigma": 0.0
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2, 3],
                "signals": [self.exog1, self.exog2]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2, 3],
                "signals": [self.interv_sig, self.exog1, self.exog2]
            },
            initial_condition=np.array([[8.25332906, 3.38215312, 5.75760548, 7.53301865, 8.27103937]]),
            timestep=0.02,
        )


class LotkaVolteraMediator(DataGenerator):

    target_idx = 1


    # Intervention signals
    interv_sig = lambda self, t: 10
    exog1 = randsig(
        max_T=300, amax=6, amin=3,
        fmax=3, rng=np.random.default_rng(13)
    )
    exog2 = randsig(
        max_T=300, amax=6, amin=3,
        fmax=3, rng=np.random.default_rng(15)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.LotkaVolteraSDE,
            model_params={
                "growth_rates": np.array([8.2617336 , 8.66335721, 8.63889329, 9.64838287, 7.68741842]),
                "capacities": np.array([13.91084807, 14.37881873, 13.72748903, 11.06953596, 14.78965454]),
                "interaction_mat": 0.5 * np.array([
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]),
                "sigma": 0.0
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2, 4],
                "signals": [self.exog1, self.exog2]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2, 4],
                "signals": [self.interv_sig, self.exog1, self.exog2]
            },
            initial_condition=np.array([[2.41352145, 2.57145249, 1.84731557, 1.93864549, 8.1382767 ]]),
            timestep=0.02,
        )


class LotkaVolteraBlockableConfound(DataGenerator):

    target_idx = 1


    # Intervention signals
    interv_sig = lambda self, t: 10
    exog1 = randsig(
        max_T=300, amax=6, amin=3,
        fmax=3, rng=np.random.default_rng(14)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.LotkaVolteraSDE,
            model_params={
                "growth_rates": np.array([11.10334803,  8.72021588, 10.64815245,  8.80041956, 11.0718007 ]),
                "capacities": np.array([11.03562434, 12.89089837, 16.61571739, 17.01295764, 14.38680003]),
                "interaction_mat": 0.5 *  np.array([
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0],
                ]),
                "sigma": 0.0
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2],
                "signals": [self.exog1]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2],
                "signals": [self.interv_sig, self.exog1]
            },
            initial_condition=np.array([[4.81406604, 7.39983144, 4.38964096, 9.59872839, 3.09459043]]),
            timestep=0.02,
        )


class MichaelisMenten1(DataGenerator):

    target_idx = 4

    interv_sig = lambda self, t: 3.0
    exog = randsig(
            max_T=1000, amax=1, amin=0,
            fmax=5, fmin=1/10, rng=np.random.default_rng(101)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.MichaelisMenten,
            model_params={
                "adjacency_matrix": 0.5 * np.array([
                    [1.0, 1.0, -1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, -1.0],
                    [0.0, -1.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ]),
                "h": 2,
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [1], "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 1], "signals": [
                    self.interv_sig, self.exog]},
            initial_condition=np.array([
                [0.82302224, 0.09089006, 0.68957824, 0.53722689, 0.99534445],
                [0.94453027, 0.06492306, 0.76213131, 0.48935338, 0.15360667]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)        
        )        


class MichaelisMenten2(DataGenerator):

    target_idx = 2

    interv_sig = lambda self, t: 4.0
    exog = randsig(
            max_T=1000, amax=1, amin=0,
            fmax=5, fmin=1/10, rng=np.random.default_rng(101)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.MichaelisMenten,
            model_params={
                "adjacency_matrix": 0.5 * np.array([
                    [1.0, 1.0, -1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, -1.0],
                    [0.0, -1.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ]),
                "h": 2,
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [1], "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [4, 1], "signals": [
                    self.interv_sig, self.exog]},
            initial_condition=np.array([
                [0.82302224, 0.09089006, 0.68957824, 0.53722689, 0.99534445],
                [0.94453027, 0.06492306, 0.76213131, 0.48935338, 0.15360667]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)        
        )


class MichaelisMenten3(DataGenerator):

    target_idx = 2

    interv_sig = lambda self, t: -1.0
    exog = randsig(
        max_T=1000, amax=1, amin=0,
        fmax=5, fmin=1/10, rng=np.random.default_rng(101)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.MichaelisMenten,
            model_params={
                "adjacency_matrix": 2 * np.array([
                    [1.0, 1.0, 1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],

                ]),
                "h": 4,
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [1], "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [4, 1], "signals": [
                    self.interv_sig, self.exog]},
            initial_condition=np.array([
                [0.82302224, 0.09089006, 0.68957824, 0.53722689, 0.99534445],
                [0.94453027, 0.06492306, 0.76213131, 0.48935338, 0.15360667]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)        
        )


class MooreSpiegel1(DataGenerator):

    target_idx = 2
    max_timestep = 0.002
    interv_sig = lambda self, t: -0.03

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.MooreSpiegel,
            model_params={"R": 100, "T": 10},
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [1], "signals": [self.interv_sig]},
            initial_condition=np.array([
                [1., 1.0, 1.0],
                [1., 1.0, 1.0]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)     
        )

    @interfere.utils.copy_doc(DataGenerator.generate_data)
    def generate_data(
        self, num_train_obs = 100,
        num_forecast_obs = 50,
        num_burn_in_states = 0
    ):
        # Stepsize must be small or forward Euler diverges.
        if self.timestep > 0.002:
            # Simulate with smaller timestep and downsample.
            new_timestep = 0.0005
            return self.generate_data_and_downsample(
                new_timestep,
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
                num_burn_in_states=num_burn_in_states,
            )
        
        else:
            return super().generate_data(
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
                num_burn_in_states=num_burn_in_states,
            )


class MooreSpiegel2(DataGenerator):

    target_idx = 2
    max_timestep = 0.002
    interv_sig = lambda self, t: -0.03

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.MooreSpiegel,
            model_params={"R": 100, "T": 40},
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [1], "signals": [self.interv_sig]},
            initial_condition=np.array([
                [1., 1.0, 1.0],
                [1., 1.0, 1.0]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)     
        )

    @interfere.utils.copy_doc(DataGenerator.generate_data)
    def generate_data(
        self, num_train_obs = 100,
        num_forecast_obs = 50,
        num_burn_in_states = 0
    ):
        # Stepsize must be small or forward Euler diverges.
        if self.timestep > 0.002:
            # Simulate with smaller timestep and downsample.
            new_timestep = 0.0005
            return self.generate_data_and_downsample(
                new_timestep,
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
                num_burn_in_states=num_burn_in_states,
            )
        
        else:
            return super().generate_data(
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
                num_burn_in_states=num_burn_in_states,
            )


class MooreSpiegel3(DataGenerator):

    target_idx = 2
    max_timestep = 0.002
    interv_sig = lambda self, t: 0.05

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.MooreSpiegel,
            model_params={"R": 100, "T": 150},
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [1], "signals": [self.interv_sig]},
            initial_condition=np.array([
                [1., 0.0, 1.0],
                [1., 0.0, 1.0]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)     
        )

    @interfere.utils.copy_doc(DataGenerator.generate_data)
    def generate_data(
        self, num_train_obs = 100,
        num_forecast_obs = 50,
        num_burn_in_states = 0
    ):
        # Stepsize must be small or forward Euler diverges.
        if self.timestep > 0.002:
            # Simulate with smaller timestep and downsample.
            new_timestep = 0.0005
            return self.generate_data_and_downsample(
                new_timestep,
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
                num_burn_in_states=num_burn_in_states,
            )
        
        else:
            return super().generate_data(
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
                num_burn_in_states=num_burn_in_states,
            )


class MutualisticPop3(DataGenerator):

    target_idx = 2

    interv_sig = lambda self, t: 400.0
    exog = randsig(
        max_T=1000, amax=300, amin=100,
        fmax=400, fmin=100, rng=np.random.default_rng(101)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.MutualisticPopulation,
            model_params={
                "alpha": np.array([0.12767238, 0.07658851, 0.17537087, 0.11031559, 0.22521503]),
                "theta": np.array([0.09350395, 0.08481134, 0.12844143, 0.1608109 , 0.11711182]),
                "adjacency_matrix": 10 * np.array([
        [0.73083525, 0.44589813, 0.01742865, 0.97795166, 0.06784397],
        [0.20935416, 0.67354246, 0.781504  , 0.38530501, 0.10222777],
        [0.32651044, 0.99212054, 0.55599965, 0.7893739 , 0.95372582],
        [0.11922454, 0.82836123, 0.56638243, 0.1140988 , 0.01417985],
        [0.03972748, 0.7415288 , 0.791318  , 0.36524593, 0.07920304]
                ]),
                "h": 3.0
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [1], "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 1], "signals": [
                    self.interv_sig, self.exog]},
            initial_condition=np.array([
                [0.82302224, 0.09089006, 0.68957824, 0.53722689, 0.99534445],
                [0.94453027, 0.06492306, 0.76213131, 0.48935338, 0.15360667]
            ]),
            start_time=0,
            timestep=0.001,
            rng = np.random.default_rng(SEED)        
        )

class MutualisticPop1(DataGenerator):

    target_idx = 3

    interv_sig = lambda self, t: 65.0
    exog = randsig(
        max_T=1000, amax=70, amin=50,
        fmax=200, fmin=20, rng=np.random.default_rng(101)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.MutualisticPopulation,
            model_params={
                "alpha": np.array([0.12767238, 0.07658851, 0.17537087, 0.11031559, 0.22521503]),
                "theta": np.array([0.09350395, 0.08481134, 0.12844143, 0.1608109 , 0.11711182]),
                "adjacency_matrix": 5 * np.array([
                    [0.0, 1.0, -1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 0.0, -1.0],
                    [1.0, 0.0, 1.0, -1.0, 0.0],
                ]),
                "h": 11
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [1], "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [4, 1], "signals": [
                    self.interv_sig, self.exog]},
            initial_condition=np.array([
                [0.82302224, 0.09089006, 0.68957824, 0.53722689, 0.99534445],
                [0.94453027, 0.06492306, 0.76213131, 0.48935338, 0.15360667]
            ]),
            start_time=0,
            timestep=0.001,
            rng = np.random.default_rng(SEED)        
        )


class MutualisticPop2(DataGenerator):

    target_idx = 4

    interv_sig = lambda self, t: 65.0
    exog = randsig(
        max_T=1000, amax=70, amin=50,
        fmax=200, fmin=20, rng=np.random.default_rng(101)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.MutualisticPopulation,
            model_params={
                "alpha": np.array([0.12767238, 0.07658851, 0.17537087, 0.11031559, 0.22521503]),
                "theta": np.array([0.09350395, 0.08481134, 0.12844143, 0.1608109 , 0.11711182]),
                "adjacency_matrix": 5 * np.array([
                    [0.0, 1.7, -1.0, 0.0, 1.0],
                    [-1.0, 1.0, 0.0, 0.9, 1.0],
                    [1.0, 1.0, 1.0, 0.0, -1.0],
                    [1.6, 0.0, 1.0, -1.0, 0.0],
                    [0.0, 10.0, 0.0, 1.0, 0.0],

                ]),
                "h": 11
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [1], "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 1], "signals": [
                    self.interv_sig, self.exog]},
            initial_condition=np.array([
                [0.82302224, 0.09089006, 0.68957824, 0.53722689, 0.99534445],
                [0.94453027, 0.06492306, 0.76213131, 0.48935338, 0.15360667]
            ]),
            start_time=0,
            timestep=0.001,
            rng = np.random.default_rng(SEED)        
        )


class OrnsteinUhlenbeck1(DataGenerator):

    target_idx = 3

    interv_sig = lambda self, t: 2.0
    exog = randsig(
        max_T=300, amax=2, amin=1,
        fmax=3, fmin=0.5, rng=np.random.default_rng(134)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.OrnsteinUhlenbeck,
            model_params={
                "mu": 0.25 * np.ones(5),
                "theta": np.array([
            [ 3.56557309,  0.40068276, -0.06194183,  0.9518984 , -0.26653658],
            [ 0.40068276,  5.42051491, -1.18463538,  0.05478063,  0.37562803],
            [-0.06194183, -1.18463538,  4.16848778,  0.48689878,  0.9497738 ],
            [ 0.9518984 ,  0.05478063,  0.48689878,  4.64490914, -0.87463056],
            [-0.26653658,  0.37562803,  0.9497738 , -0.87463056,  5.19931559]
            ]),
                "sigma": np.zeros((5, 5)),
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2],
                "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2],
                "signals": [self.interv_sig, self.exog]
            },
            initial_condition=np.array([
                [-1.13237842,  0.85593583,  2.11953355,  1.47173214,  0.6987275],
                [-2.47019708, -0.75337922, -0.29746292,  1.1320844 ,  1.91501353]
            ]),
            start_time=0,
            timestep=0.025,
            rng = np.random.default_rng(SEED)
        )

class OrnsteinUhlenbeck2(DataGenerator):

    target_idx = 3

    interv_sig = lambda self, t: 2.0
    exog = randsig(
        max_T=300, amax=1, amin=-1,
        fmax=3, fmin=0.5, rng=np.random.default_rng(134)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.OrnsteinUhlenbeck,
            model_params={
                "mu": 0.25 * np.ones(5),
                "theta": np.array([
            [ 7.09073381e-02, -4.35446589e-02, -3.33116518e-03,
         6.21107254e-02,  4.87968309e-06],
       [-4.35446589e-02,  2.00812406e-01,  4.84760402e-03,
        -2.79680744e-02, -4.44680777e-02],
       [-3.33116518e-03,  4.84760402e-03,  8.38429368e-02,
        -5.11284976e-02,  6.62722342e-02],
       [ 6.21107254e-02, -2.79680744e-02, -5.11284976e-02,
         1.67914192e-01, -5.82372371e-02],
       [ 4.87968309e-06, -4.44680777e-02,  6.62722342e-02,
        -5.82372371e-02,  1.50686365e-01]
            ]),
                "sigma": np.zeros((5, 5)),
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [1],
                "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 1],
                "signals": [self.interv_sig, self.exog]
            },
            initial_condition=np.array([
                [-1.13237842,  0.85593583,  2.11953355,  1.47173214,  0.6987275],
                [-2.47019708, -0.75337922, -0.29746292,  1.1320844 ,  1.91501353]
            ]),
            start_time=0,
            timestep=0.025,
            rng = np.random.default_rng(SEED)
        )


class OrnsteinUhlenbeck3(DataGenerator):

    target_idx = 3

    interv_sig = lambda self, t: 2.0
    exog = randsig(
        max_T=300, amax=1, amin=-1,
        fmax=3, fmin=0.5, rng=np.random.default_rng(139)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.OrnsteinUhlenbeck,
            model_params={
                "mu": 0.25 * np.ones(5),
                "theta": np.array([
                    [ 7.8851358 ,  0.95810051, -0.13271091, -1.0334837 , -1.38060408],
                    [ 0.95810051,  4.92767994, -0.35044033,  1.72013998,  0.89830353],
                    [-0.13271091, -0.35044033,  5.47426675,  0.77311291,  1.51043686],
                    [-1.0334837 ,  1.72013998,  0.77311291,  6.5859663 , -1.33664886],
                    [-1.38060408,  0.89830353,  1.51043686, -1.33664886,  7.8666837 ]
                ]),
                "sigma": np.zeros((5, 5)),
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [1],
                "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 1],
                "signals": [self.interv_sig, self.exog]
            },
            initial_condition=np.array([
                [ 1.54670353,  2.25712311, -1.78229183, -0.08259701,  2.84710893],
                [ 0.6928484 , -2.09602794, -0.69285883, -1.87164538,  1.95915625]
            ]),
            start_time=0,
            timestep=0.025,
            rng = np.random.default_rng(SEED)
        )


class PlantedTank1(DataGenerator):

    target_idx = 3

    interv_sig = lambda self, t: 2.0
    exog = randsig(
            max_T=24 * 3600 * 7000, amax=3, amin=2,
            fmax=1/2000, fmin=1/10000, rng=np.random.default_rng(117)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.PlantedTankNitrogenCycle,
            model_params={"sigma": 0.0},
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2], "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [1, 2], "signals": [
                    self.interv_sig, self.exog]},
            initial_condition=np.array([
                [0.66158597, 0.8012904 , 0.19920669, 0.96602119],
                [0.8102566 ,  0.10224813, -0.35046573, 0.318422]
            ]),
            start_time=0,
            timestep=60*60*24,
            rng = np.random.default_rng(SEED)        
        )


class PlantedTank2(DataGenerator):

    target_idx = 3

    interv_sig = lambda self, t: 2.5
    exog = randsig(
        max_T=24 * 3600 * 7000, amax=3, amin=2,
        fmax=1/20000, fmin=1/100000, rng=np.random.default_rng(101)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.PlantedTankNitrogenCycle,
            model_params={"sigma": 0.0},
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [1], "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 1], "signals": [
                    self.interv_sig, self.exog]},
            initial_condition=np.array([
                [0.66158597, 0.8012904 , 0.19920669, 0.96602119],
                [0.8102566 ,  0.10224813, -0.35046573, 0.318422]
            ]),
            start_time=0,
            timestep=60*60*24, # 1 Day in seconds.
            rng = np.random.default_rng(SEED)        
        )


class PlantedTank3(DataGenerator):

    target_idx = 2

    interv_sig = lambda self, t: 1.0
    exog = randsig(
            max_T=24 * 3600 * 7000, amax=3, amin=2,
            fmax=1/20000, fmin=1/100000, rng=np.random.default_rng(107)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.PlantedTankNitrogenCycle,
            model_params={
                "nitrate_bact_eff_gamma2": 0.00000003
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [1], "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 1], "signals": [
                    self.interv_sig, self.exog]},
            initial_condition=np.array([
                [0.66158597, 0.8012904 , 0.19920669, 0.96602119],
                [0.8102566 ,  0.10224813, -0.35046573, 0.318422]
            ]),
            start_time=0,
            timestep=60*60*24,
            rng = np.random.default_rng(SEED)  
        )


class Rossler1(DataGenerator):

    target_idx=1

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Rossler,
            model_params={},
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 4.0},
            initial_condition=np.array([
                [0.66158597, 0.8012904 , 0.19920669],
                [0.0 ,  0.10224813, -0.35046573]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)        
        )

    @interfere.utils.copy_doc(DataGenerator.generate_data)
    def generate_data(
        self, num_train_obs = 100,
        num_forecast_obs = 50,
        num_burn_in_states = 0
    ):
        # Stepsize must be small or forward Euler diverges.
        if self.timestep > 0.002:
            # Simulate with smaller timestep and downsample.
            new_timestep = 0.001
            return self.generate_data_and_downsample(
                new_timestep,
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
                num_burn_in_states=num_burn_in_states,
            )
        
        else:
            return super().generate_data(
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
            )


class Rossler2(DataGenerator):

    target_idx=1

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Rossler,
            model_params={"b": 0.5},
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 2, "constants": 10.0},
            initial_condition=np.array([
                [0.66158597, 0.8012904 , 0.19920669],
                [0.8102566 ,  0.10224813, -0.35046573]
            ]),
            start_time=0,
            timestep=0.05,
            rng = np.random.default_rng(SEED)        
        )


    @interfere.utils.copy_doc(DataGenerator.generate_data)
    def generate_data(
        self, num_train_obs = 100,
        num_forecast_obs = 50,
        num_burn_in_states = 0
    ):
        # Stepsize must be small or forward Euler diverges.
        if self.timestep > 0.002:
            # Simulate with smaller timestep and downsample.
            new_timestep = 0.001
            return self.generate_data_and_downsample(
                new_timestep,
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
                num_burn_in_states=num_burn_in_states,
            )
        
        else:
            return super().generate_data(
                num_train_obs=num_train_obs,
                num_forecast_obs=num_forecast_obs,
            )


class Thomas1(DataGenerator):

    target_idx=1

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Thomas,
            model_params={},
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 0, "constants": 3.0},
            initial_condition=np.array([
                [0.66158597, 0.8012904 , 0.19920669],
                [0.8102566 ,  0.10224813, -0.35046573]
            ]),
            start_time=0,
            timestep=0.1,
            rng = np.random.default_rng(SEED)        
        )


class Thomas2(DataGenerator):

    target_idx=2

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.Thomas,
            model_params={},
            do_intervention_type=interfere.PerfectIntervention,
            do_intervention_params={"intervened_idxs": 1, "constants": 3.0},
            initial_condition=np.array([
                [0.66158597, 0.8012904 , 0.19920669],
                [0.8102566 ,  0.10224813, -0.35046573]
            ]),
            start_time=0,
            timestep=0.1,
            rng = np.random.default_rng(SEED)        
        )


class VARMA1SpatiotempChaos(DataGenerator):

    target_idx = 9

    interv_sig = lambda self, t: 0.0
    exog = randsig(
            max_T=11000, amax=1, amin=-1,
            fmax=2, fmin=0.5, rng=np.random.default_rng(111)
        )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.VARMADynamics,
            model_params={
                "phi_matrices": np.array([
                    [[-8.22567594e-01,  1.53146164e+00, -7.76564544e-01,
                        -1.98575390e-01, -4.05473100e-01,  4.30871557e-01,
                        5.00184816e-02,  1.09480417e+00, -4.49578612e-01,
                        -1.85126401e-01],
                        [-6.77832363e-02,  6.51175375e-01, -3.72316063e-01,
                        -2.80106956e-01, -1.74793727e-01,  2.66750787e-01,
                        -2.19511273e-01,  7.12887316e-01, -1.57809809e-01,
                        -3.85194047e-01],
                        [-2.90014146e-02,  3.38541653e-01,  7.12068365e-02,
                        -4.90755012e-01, -2.94920059e-01,  1.78923633e-01,
                        -4.44610282e-01,  7.91517347e-01, -2.53216616e-01,
                        -1.70283250e-01],
                        [-5.79615280e-02,  1.65624855e-01, -3.43983659e-02,
                        -4.24594391e-01, -5.32060476e-01,  2.96127958e-01,
                        -5.19283742e-01,  8.14860340e-01,  4.41977792e-02,
                        -1.11633720e-01],
                        [ 9.86126937e-03,  2.45382231e-01,  6.33490635e-02,
                        -2.97236099e-01, -5.81342637e-01,  2.49170296e-01,
                        -4.71460623e-01,  6.36819345e-01, -2.09313013e-01,
                        -9.24980153e-02],
                        [ 6.59769352e-02,  3.58596445e-01,  4.40647729e-02,
                        -4.87833020e-01, -4.31006951e-01,  1.98033467e-01,
                        -3.83965921e-01,  7.64330964e-01, -2.02207500e-01,
                        -2.38772856e-01],
                        [-3.74258368e-02,  1.32382118e-01,  2.22320599e-02,
                        -4.78237683e-01, -4.96095064e-01,  4.06221331e-01,
                        -4.42628916e-01,  7.50854074e-01, -1.32537964e-01,
                        -1.04297366e-01],
                        [ 1.88747687e-02,  2.69769427e-01, -1.66553380e-01,
                        -5.04649277e-01, -3.19823568e-01,  2.64199799e-01,
                        -3.39312834e-01,  8.07455791e-01,  6.53318011e-02,
                        -2.80452162e-01],
                        [-1.38798083e-01,  5.14346176e-01, -1.21126446e-01,
                        -4.95130765e-01, -1.82968201e-01,  4.52385311e-01,
                        -1.43563680e-01,  4.93813240e-01, -1.68785332e-01,
                        -2.63469404e-01],
                        [-8.16259722e-01,  1.71587484e+00, -7.34142000e-01,
                        -1.42556515e-01, -5.19691788e-01,  3.98089229e-01,
                        2.10213644e-01,  1.00038709e+00, -4.51170219e-01,
                        -2.75517459e-01]],

                    [[-1.85979620e-01,  6.33302523e-01, -2.18964180e-01,
                        9.43999678e-01, -1.06598273e-01, -8.71994582e-01,
                        -1.47736161e-01,  2.31593883e-01,  4.85380829e-01,
                        4.71160328e-03],
                        [ 8.47800911e-03, -2.95816886e-02,  2.95718120e-01,
                        4.69971885e-01, -1.43294632e-01, -5.47199565e-01,
                        1.74657783e-01, -6.73875248e-02, -9.31689068e-02,
                        1.13446248e-01],
                        [-2.62800788e-03, -3.57721007e-01,  1.66386354e-01,
                        6.60041654e-01, -1.46334205e-01, -4.05798084e-01,
                        1.32979077e-01, -2.36211633e-01, -7.19353407e-02,
                        2.71391501e-01],
                        [-1.31431207e-01, -6.11367794e-01,  3.54021169e-01,
                        5.83495254e-01, -1.62084410e-01, -1.94920288e-01,
                        1.19128493e-01, -2.99520451e-01, -2.18223980e-01,
                        4.88260565e-01],
                        [-9.75939203e-02, -5.46364053e-01,  2.80937852e-01,
                        4.00154638e-01, -1.88573862e-01,  1.00274358e-02,
                        3.77203572e-01, -4.69804006e-01, -3.67892839e-01,
                        4.66852061e-01],
                        [-9.99981590e-02, -6.96887416e-01,  2.23228645e-01,
                        4.80722420e-01, -2.55150086e-02,  3.72311991e-02,
                        3.24986645e-01, -5.81398951e-01, -1.06842441e-01,
                        4.19855189e-01],
                        [-8.67672531e-02, -5.63304522e-01,  8.01355152e-02,
                        7.38505315e-01, -1.44968274e-01, -8.83912338e-02,
                        1.86154326e-01, -3.83035828e-01, -1.71100600e-01,
                        4.21694718e-01],
                        [-7.79137421e-02, -3.79943879e-01,  3.29691217e-01,
                        5.10701924e-01, -2.29839366e-01, -1.61510633e-01,
                        -5.22140910e-04, -3.36525262e-01,  7.53699861e-02,
                        3.11316781e-01],
                        [-5.45918414e-02, -8.64724385e-02, -5.89586239e-02,
                        5.85144187e-01, -1.91199531e-01, -3.21501274e-01,
                        8.81681770e-02,  6.10854995e-02,  1.43950277e-02,
                        1.91698413e-01],
                        [-1.69918066e-01,  6.33063207e-01, -4.45905729e-02,
                        9.99799404e-01, -3.51417074e-01, -9.22530995e-01,
                        7.26810447e-02,  1.23356997e-01,  4.59985129e-01,
                        -1.61685925e-02]]]),
                "theta_matrices": np.array([
                    [[-0.01937861, -1.40297266, -0.06245954, -0.02323428,
                        0.0235188 , -0.52123149, -0.05612515, -1.60018851,
                        1.19732026,  0.27065969],
                        [-0.04035818, -0.43052108,  0.01562953,  0.12574238,
                        0.07874397, -0.37590511, -0.08134999, -0.58674095,
                        0.2832178 ,  0.17948791],
                        [-0.03810049, -0.40740769,  0.06360897,  0.2198333 ,
                        0.16251351, -0.39855474, -0.01904092, -0.6530906 ,
                        0.19674952,  0.18095096],
                        [-0.03359532, -0.31980367,  0.03449592,  0.1991286 ,
                        0.25479637, -0.34493496,  0.01817239, -0.57306288,
                        0.146024  ,  0.14664465],
                        [-0.02218604, -0.44317824,  0.05439248,  0.18003508,
                        0.27768926, -0.34554548,  0.0285065 , -0.67018076,
                        0.25233191,  0.14462599],
                        [-0.0934728 , -0.40570531, -0.00273427,  0.11376973,
                        0.20651947, -0.20010243,  0.11557645, -0.74879693,
                        0.20684444,  0.2111111 ],
                        [-0.09542369, -0.35357053,  0.03915915,  0.18433727,
                        0.20645126, -0.41279622,  0.03767577, -0.56929259,
                        0.249837  ,  0.19120006],
                        [-0.07932283, -0.32320889,  0.07571131,  0.19366307,
                        0.18289911, -0.45519083, -0.09760803, -0.55043321,
                        0.25638428,  0.14966224],
                        [-0.00451479, -0.50054479,  0.01418209,  0.10917693,
                        0.1050911 , -0.41775471, -0.1038681 , -0.50661686,
                        0.33551537,  0.1375448 ],
                        [-0.09725128, -1.25434395, -0.1631592 , -0.07038576,
                        0.16143039, -0.53860592,  0.05554345, -1.46693243,
                        1.0246715 ,  0.3448464 ]],

                    [[-0.49444556,  0.10375126, -0.30857068,  0.15633412,
                        0.83537704,  0.06283299, -0.0780449 , -0.89155664,
                        0.60848489,  0.20747319],
                        [-0.21784139,  0.02565041, -0.04558072, -0.05298373,
                        0.26974464,  0.16484666, -0.06376414, -0.30567509,
                        0.31555401,  0.08723757],
                        [-0.14392357,  0.26601489,  0.06244015, -0.15548385,
                        0.19103852,  0.06452093, -0.09489329, -0.07558061,
                        0.2907206 , -0.05907708],
                        [-0.0723089 ,  0.21471905,  0.10886172, -0.12523671,
                        0.06948565,  0.08934593, -0.15191143,  0.03805515,
                        0.23974672, -0.10435403],
                        [-0.08427363,  0.22606296,  0.06741797, -0.18380392,
                        0.15744134,  0.09303937, -0.09741852, -0.01870524,
                        0.2679603 , -0.10302689],
                        [-0.13180626,  0.25972952,  0.10379222, -0.07273589,
                        0.17409492, -0.07490287, -0.19752365,  0.06349131,
                        0.21934322, -0.04932973],
                        [-0.00730032,  0.06499061,  0.00359724, -0.08417728,
                        0.13212261,  0.07665775, -0.01071523, -0.009403  ,
                        0.23013631, -0.11216423],
                        [-0.11295223,  0.10366513,  0.09369875, -0.05687349,
                        0.26337525,  0.0865514 ,  0.0152967 , -0.25958896,
                        0.35237178, -0.08414   ],
                        [-0.1836445 , -0.00398083, -0.04440542, -0.03823344,
                        0.28305817,  0.2137743 , -0.0369061 , -0.34998129,
                        0.27740858,  0.07370178],
                        [-0.50045093, -0.01208872, -0.38064556,  0.10320213,
                        0.77419262,  0.22129293, -0.02921978, -0.9169595 ,
                        0.61419277,  0.28029008]],

                    [[ 0.18045063,  0.20907762,  0.88569217, -0.43666674,
                        -0.22958766, -0.06206391, -0.18031736,  0.33711588,
                        0.88907047, -0.863676  ],
                        [ 0.16336101,  0.00615319, -0.0363368 , -0.19479188,
                        -0.08555033,  0.02354517, -0.14403862,  0.51619935,
                        0.17672803, -0.32046224],
                        [-0.06495671,  0.14532208, -0.17198729, -0.34036496,
                        0.00948189,  0.35384384, -0.06988188,  0.34740214,
                        -0.12310483, -0.02500279],
                        [ 0.09105773,  0.05185448, -0.14192136, -0.19703747,
                        -0.19588605,  0.27356002, -0.12450841,  0.48341252,
                        -0.15898339, -0.14023397],
                        [ 0.20986185, -0.02652572, -0.04255398, -0.21515668,
                        0.01909063,  0.1073224 , -0.16115045,  0.43716124,
                        -0.00717714, -0.30414828],
                        [ 0.05536861,  0.09864658, -0.05679519, -0.50263733,
                        -0.00307957,  0.48045228, -0.20346513,  0.44971775,
                        0.04952851, -0.2314951 ],
                        [-0.01271917,  0.39576738, -0.14728319, -0.28690946,
                        0.11386424,  0.09949462, -0.15672682,  0.41677498,
                        -0.23330107, -0.1169405 ],
                        [ 0.06750606,  0.17805581, -0.04860101, -0.23548557,
                        -0.01687164,  0.24000338, -0.25689052,  0.51692769,
                        -0.03385882, -0.22819766],
                        [ 0.17995369,  0.06809188,  0.07583467, -0.26244196,
                        -0.17792351, -0.03301446, -0.12604723,  0.7067164 ,
                        0.04964482, -0.34101772],
                        [ 0.33510634,  0.31827004,  0.7734377 , -0.61190236,
                        -0.26594593, -0.1173287 , -0.52939879,  0.83693306,
                        0.73400108, -0.95488648]]]),
                "sigma": np.array([
                    [0.7232, 0.1361, 0.062 , 0.0815, 0.0668, 0.0788, 0.0569, 0.0746,
                        0.0602, 0.0635],
                    [0.1361, 0.124 , 0.0815, 0.0668, 0.0788, 0.0569, 0.0746, 0.0602,
                        0.0635, 0.0599],
                    [0.062 , 0.0815, 0.1336, 0.0788, 0.0569, 0.0746, 0.0602, 0.0635,
                        0.0599, 0.0707],
                    [0.0815, 0.0668, 0.0788, 0.1138, 0.0746, 0.0602, 0.0635, 0.0599,
                        0.0707, 0.0617],
                    [0.0668, 0.0788, 0.0569, 0.0746, 0.1204, 0.0635, 0.0599, 0.0707,
                        0.0617, 0.028 ],
                    [0.0788, 0.0569, 0.0746, 0.0602, 0.0635, 0.1198, 0.0707, 0.0617,
                        0.028 , 0.054 ],
                    [0.0569, 0.0746, 0.0602, 0.0635, 0.0599, 0.0707, 0.1234, 0.028 ,
                        0.054 , 0.066 ],
                    [0.0746, 0.0602, 0.0635, 0.0599, 0.0707, 0.0617, 0.028 , 0.108 ,
                        0.066 , 0.063 ],
                    [0.0602, 0.0635, 0.0599, 0.0707, 0.0617, 0.028 , 0.054 , 0.066 ,
                        0.126 , 0.053 ],
                    [0.0635, 0.0599, 0.0707, 0.0617, 0.028 , 0.054 , 0.066 , 0.063 ,
                        0.053 , 0.0708]]),
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2],
                "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2],
                "signals": [self.interv_sig, self.exog]
            },
            initial_condition=np.array([
                [-2.14616   ,  2.50546601, -2.25526292,  1.0125853 , -1.71726825,
         0.86319739,  2.49925294,  2.01354154,  0.68934069, -0.03081311],
                [-0.4307467 ,  1.22192879,  1.57553691,  0.91852625, -0.14034144,
         1.43745407, -0.91566823, -1.44399245, -2.70617114, -1.81966176],
            ]),
            start_time=0,
            timestep=1,
            rng = np.random.default_rng(SEED)
        )


class VARMA2ChaoticBrownian(DataGenerator):


    target_idx = 2

    interv_sig = lambda self, t: 1.0
    exog = randsig(
            max_T=11000, amax=1, amin=-1,
            fmax=2, fmin=0.5, rng=np.random.default_rng(111)
        )
    
    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.VARMADynamics,
            model_params={
            "phi_matrices": np.array([
                [[-5.83933408e-01,  3.99005737e-03,  9.32627185e-01,
                    -4.62156300e-02, -3.67625281e-02, -3.05486622e-01,
                    -3.83705592e-01,  6.08579970e-01,  1.93105113e-01,
                    -1.37668914e-01],
                    [ 1.95979101e-01, -3.43092959e-01,  2.64403946e-01,
                    6.92146759e-02,  8.99118120e-02, -5.30917098e-02,
                    2.77887372e-02, -1.21117511e-01,  5.38357048e-02,
                    -2.75716400e-01],
                    [-7.66287569e-02,  2.37386262e-02, -7.33353832e-02,
                    -2.56865947e-01, -2.00658630e-01,  1.24508171e-01,
                    1.42909505e-01, -1.15706080e-03,  3.69247817e-01,
                    2.55876885e-04],
                    [ 6.97087156e-02, -4.58042641e-02,  1.33103609e-01,
                    -1.84141265e-01,  2.31949549e-01,  1.47087026e-01,
                    2.43731081e-02, -2.62567356e-01, -9.04124430e-02,
                    -8.21354031e-02],
                    [ 1.16153347e-01,  2.93595609e-03,  2.50869432e-01,
                    -1.09785784e-01,  1.58785819e-01,  1.42518277e-01,
                    6.30190913e-02, -2.13666674e-01, -2.59046225e-01,
                    -9.67992615e-02],
                    [-1.18739272e-03,  2.22442729e-02,  3.53841444e-02,
                    -1.59960543e-01, -1.01284080e-01,  2.24427231e-02,
                    3.10417535e-01, -1.67481369e-01,  1.28366685e-01,
                    -3.36808840e-02],
                    [ 9.55611187e-02,  1.02315351e-01,  1.96794543e-02,
                    -1.92875170e-01,  2.23834983e-01,  1.46535458e-01,
                    8.18916944e-02, -2.14831704e-01, -1.76368190e-01,
                    -9.27698617e-02],
                    [-1.11809599e-01,  1.68292535e-01,  3.80112999e-02,
                    -1.46931671e-02,  2.28369503e-01, -1.23536286e-02,
                    4.05130796e-02,  2.99970768e-02, -4.04803877e-01,
                    1.53015796e-01],
                    [ 4.87950686e-03, -2.64891818e-01,  2.70935608e-01,
                    -2.02173561e-03, -1.62948113e-02, -5.38113705e-02,
                    1.93180715e-01, -2.53283139e-02,  2.33962517e-02,
                    -9.68930106e-02],
                    [-5.48108929e-01,  9.28100386e-02,  1.03203535e+00,
                    -6.02241407e-02, -2.53963287e-01, -3.93807336e-01,
                    -3.39020098e-01,  7.05402010e-01,  3.14758122e-01,
                    -1.76523564e-01]],

                [[ 3.32246317e-01, -2.12146466e-01, -4.18732545e-01,
                    -3.19795019e-01,  7.87962348e-02, -1.00446240e-01,
                    -2.00737695e-01,  8.29735014e-01,  2.75701172e-01,
                    -4.42302504e-01],
                    [ 1.87917944e-01, -1.13179033e-01,  2.53063267e-01,
                    -4.42472696e-02,  2.95745613e-02, -2.14008136e-02,
                    -2.21099795e-01, -1.66200342e-01,  2.80209295e-01,
                    -1.87076317e-01],
                    [-1.02452995e-01,  5.90055711e-02,  1.38286423e-01,
                    -7.82799591e-02, -3.51538748e-03,  9.51047394e-02,
                    -2.26051746e-01, -2.26902455e-01,  2.32228498e-01,
                    1.33944471e-01],
                    [ 2.71425167e-02, -3.75083190e-02,  2.80991968e-01,
                    1.97999297e-01, -1.38956982e-01, -3.45483226e-01,
                    -1.15814196e-01, -1.51036544e-01,  3.57211423e-01,
                    -5.01123881e-02],
                    [-4.95258094e-02, -1.09594704e-01,  1.08026427e-01,
                    3.29954732e-01, -1.11154885e-02, -1.91384969e-01,
                    -2.98641798e-01, -1.20672084e-01,  2.51837991e-01,
                    5.22225956e-02],
                    [ 9.28501210e-02, -1.23240225e-01,  3.76381633e-01,
                    2.83827008e-01, -2.00640203e-01, -1.98409713e-01,
                    -2.22327360e-01, -2.59751311e-01,  3.16424594e-01,
                    -8.63982354e-02],
                    [-6.13619216e-02, -2.30258151e-01,  3.06480182e-01,
                    2.29480675e-01, -7.02910317e-02, -1.81451360e-01,
                    -1.16442307e-01, -8.40594302e-02,  9.17946501e-02,
                    6.17018326e-02],
                    [-2.08328452e-01,  3.47928945e-02, -1.38304201e-01,
                    1.86266678e-01, -9.65917948e-02, -7.34197876e-02,
                    -1.22554110e-01, -8.04303615e-02,  2.73876560e-01,
                    2.00396383e-01],
                    [-1.21792225e-01, -1.81990878e-02, -6.96181787e-02,
                    1.54638617e-02, -6.29865320e-02, -1.07212092e-01,
                    1.30421512e-02, -1.10032524e-01,  3.46756865e-01,
                    1.35992016e-01],
                    [ 2.11887564e-01, -9.74475579e-03, -6.57828937e-01,
                    -4.11330782e-01,  3.06793741e-01, -8.60871986e-02,
                    -1.37735144e-01,  7.71064996e-01,  1.61272913e-01,
                    -2.73319957e-01]]]),
            "theta_matrices": np.array([
                [[-0.5083923 , -0.97934722, -0.59067899, -1.09153945,
                    -0.31723178, -0.05986228, -0.30616146, -0.45257932,
                    -0.18132299,  0.50794833],
                    [-0.0576423 ,  0.00333438, -0.04189382, -0.20291153,
                    -0.15747537,  0.01944783, -0.12068506, -0.2087106 ,
                    -0.06926577,  0.06153254],
                    [ 0.05489552,  0.02111307, -0.14078398, -0.0529827 ,
                    -0.05197807, -0.06822097, -0.03054411, -0.14121021,
                    -0.1470776 , -0.04687407],
                    [ 0.04146696,  0.03642441,  0.02005099,  0.08945951,
                    -0.00471406,  0.02081045,  0.0028524 ,  0.05096156,
                    0.01563   , -0.03715595],
                    [ 0.03392461,  0.07710688,  0.04084528,  0.00964977,
                    0.00807284,  0.05290119, -0.0621836 , -0.04415987,
                    -0.07070932, -0.03511541],
                    [-0.01971554,  0.09510644, -0.01439322, -0.06924633,
                    -0.07192216, -0.04696986, -0.15368768, -0.06140946,
                    -0.07194455,  0.03465237],
                    [ 0.01253709,  0.03365361, -0.00561583,  0.01926504,
                    -0.01626456,  0.11460203,  0.02204501, -0.04485686,
                    -0.00289171, -0.03162528],
                    [-0.02971889, -0.05558858,  0.09187279,  0.02813165,
                    -0.01901941, -0.03970797, -0.01152096, -0.03904066,
                    -0.02681061,  0.03359907],
                    [-0.01258992, -0.08488836, -0.18627784, -0.10366981,
                    -0.04774095,  0.03143374, -0.0509464 , -0.07771445,
                    0.02328135,  0.00330402],
                    [-0.52078892, -0.91944769, -0.59233451, -1.16582976,
                    -0.38290876, -0.145108  , -0.32451373, -0.48655862,
                    -0.21993161,  0.51929225]],

                [[-0.20752286, -0.11925652,  0.02400374, -0.3447381 ,
                    0.27142559,  0.15952149,  0.55576314,  0.34347334,
                    0.6808926 , -0.14643719],
                    [-0.03844294,  0.04437865, -0.00645388,  0.18198072,
                    0.16185537,  0.06562146,  0.03634063,  0.05989468,
                    0.03189052, -0.00510706],
                    [-0.06328983,  0.00431713, -0.11887808, -0.13938564,
                    0.02591854, -0.10284711, -0.10228629, -0.01315786,
                    -0.02153372,  0.08100532],
                    [-0.00985417, -0.04393588,  0.02220255, -0.06415942,
                    -0.03249706,  0.03027141,  0.04244173,  0.01736867,
                    0.03898308,  0.00431735],
                    [-0.05607679, -0.02579419, -0.06569404,  0.07667167,
                    0.02575559, -0.06450177,  0.10127653, -0.10500665,
                    0.01826289,  0.04587937],
                    [ 0.01225912, -0.02650018,  0.08879548,  0.09959765,
                    0.07158755,  0.07557702,  0.15198024,  0.10066299,
                    0.01738737, -0.01949409],
                    [-0.09047587,  0.02623375, -0.00729343,  0.03191157,
                    0.09019576, -0.03761719,  0.0393641 , -0.05673668,
                    0.09141834,  0.08706423],
                    [-0.04538601,  0.07869652,  0.08201615,  0.01079721,
                    -0.00275159,  0.06888583, -0.05430875, -0.01121112,
                    -0.01905778,  0.04851025],
                    [-0.02016775, -0.09004448, -0.07506339, -0.07673776,
                    0.0586852,  0.01764962,  0.0728478 ,  0.0433706 ,
                    0.15285229, -0.02034979],
                    [-0.20230409, -0.15342485, -0.03207973, -0.44317144,
                    0.23136883,  0.07756882,  0.47048785,  0.23927052,
                    0.48826517, -0.12657132]],

                [[-0.84463111, -0.75957839,  0.36082502, -0.81390851,
                    0.57064037, -1.09070723,  0.05966974,  0.24620799,
                    1.19518734,  0.46095693],
                    [-0.17209489,  0.0415258 , -0.23686101,  0.05658422,
                    -0.0432281 ,  0.06058207, -0.04563657, -0.0611321 ,
                    0.09537442,  0.11959502],
                    [ 0.01187896, -0.04412306, -0.20513227, -0.14861293,
                    -0.11228355,  0.13240297, -0.01304757, -0.0367349 ,
                    0.02579973, -0.00623007],
                    [ 0.04386006, -0.0227546 , -0.01384728,  0.00722947,
                    0.05545264, -0.06122026,  0.00937725, -0.18425059,
                    -0.26680888, -0.0156157 ],
                    [-0.12369423,  0.11255302, -0.14990424,  0.02379842,
                    -0.05609456,  0.09146435,  0.05048998,  0.09750891,
                    -0.01627506,  0.11254491],
                    [-0.11749451, -0.17666745,  0.11909908, -0.09720342,
                    -0.12757157,  0.07152022, -0.06869978, -0.06582282,
                    0.02736357,  0.12615973],
                    [-0.13400206, -0.14327511, -0.08526781,  0.07306699,
                    0.0545118 , -0.0626452 , -0.19246182,  0.33923238,
                    0.06135424,  0.13645331],
                    [ 0.04148287, -0.04069544,  0.11710634,  0.2806305 ,
                    0.12298064,  0.2000127 ,  0.13834179,  0.08216194,
                    0.20272613, -0.07595629],
                    [-0.08282678,  0.07289228,  0.01576949, -0.26777194,
                    0.0606841 , -0.06898593, -0.36465448,  0.09245589,
                    -0.04625554,  0.08160276],
                    [-0.75152527, -0.81115822,  0.14163234, -0.58069301,
                    0.26599984, -1.14624996,  0.11557861,  0.11859273,
                    1.14423699,  0.3669352 ]]]),
            "sigma": np.array([
                [ 0.8692,  0.0153,  0.1089, -0.0068,  0.0634,  0.0764, -0.0238,
                    0.0646,  0.0369,  0.0811],
                [ 0.0153,  0.2178, -0.0068,  0.0634,  0.0764, -0.0238,  0.0646,
                    0.0369,  0.0811, -0.026 ],
                [ 0.1089, -0.0068,  0.1268,  0.0764, -0.0238,  0.0646,  0.0369,
                    0.0811, -0.026 ,  0.0451],
                [-0.0068,  0.0634,  0.0764, -0.0476,  0.0646,  0.0369,  0.0811,
                    -0.026 ,  0.0451,  0.028 ],
                [ 0.0634,  0.0764, -0.0238,  0.0646,  0.0738,  0.0811, -0.026 ,
                    0.0451,  0.028 ,  0.0441],
                [ 0.0764, -0.0238,  0.0646,  0.0369,  0.0811, -0.052 ,  0.0451,
                    0.028 ,  0.0441,  0.0649],
                [-0.0238,  0.0646,  0.0369,  0.0811, -0.026 ,  0.0451,  0.056 ,
                    0.0441,  0.0649, -0.0238],
                [ 0.0646,  0.0369,  0.0811, -0.026 ,  0.0451,  0.028 ,  0.0441,
                    0.1298, -0.0238,  0.0603],
                [ 0.0369,  0.0811, -0.026 ,  0.0451,  0.028 ,  0.0441,  0.0649,
                    -0.0238,  0.1206,  0.0352],
                [ 0.0811, -0.026 ,  0.0451,  0.028 ,  0.0441,  0.0649, -0.0238,
                    0.0603,  0.0352,  0.1128]]),
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [9],
                "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 9],
                "signals": [self.interv_sig, self.exog]
            },
            initial_condition=np.array([
                [ 0.79693014,  1.65627622, -0.15361451, -1.80763549, -2.47864785,
                -1.53604023, -1.8893948 ,  0.8139425 ,  2.92361533, -2.84296333],
                [ 0.70528409,  1.61891942,  0.14397764, -0.04500315, -0.985135  ,
                -1.95080453,  1.20280778,  2.99636559,  1.46237589,  2.21972081]
            ]),
            start_time=0,
            timestep=1,
            rng = np.random.default_rng(SEED)
        )


class VARMA3LotkaVoltera(DataGenerator):

    target_idx = 6

    interv_sig = lambda self, t: 1.0
    exog = randsig(
            max_T=11000, amax=1, amin=-1,
            fmax=2, fmin=0.5, rng=np.random.default_rng(115)
        )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.VARMADynamics,
            model_params={
                "phi_matrices": np.array([
                    [[ 6.75452022e-02, -4.51869254e-02,  5.35868087e-02,
                        -2.15531791e-02, -7.74568279e-02, -3.41994819e-02,
                        3.50651412e-02,  5.74738368e-02,  7.67324502e-03,
                        -2.97475197e-02],
                        [-1.42999602e-02,  5.46534529e-02, -1.68330227e-02,
                        1.70281159e-02, -2.08606062e-02, -5.80788227e-02,
                        -6.62563888e-03, -3.13758057e-02, -1.57989108e-02,
                        -3.11957687e-02],
                        [ 3.45177386e-02,  1.58785814e-02,  4.71412430e-02,
                        -3.28207284e-02, -3.78430136e-03,  5.01313710e-03,
                        7.81155251e-02,  5.94269072e-03,  2.62188297e-02,
                        5.19060599e-02],
                        [-3.37165751e-02,  5.10934106e-03,  4.75640503e-02,
                        1.00621765e-01, -1.19970853e-01,  2.46751695e-02,
                        -1.18594181e-01,  8.33920863e-02,  5.63359388e-02,
                        -1.30909470e-02],
                        [ 4.30253541e-02, -1.28475518e-02, -2.38859533e-03,
                        -2.93950920e-02,  8.99684183e-02, -2.44225272e-02,
                        4.85757693e-03, -3.33148969e-02,  7.81554408e-03,
                        -3.29214520e-02],
                        [-7.15288508e-02, -8.12875780e-03, -1.19736245e-02,
                        2.09589574e-02,  2.85803010e-02,  6.68495459e-02,
                        -4.77122408e-02,  5.32269116e-02,  3.14694260e-02,
                        7.82352289e-02],
                        [ 6.32022771e-02, -4.99153868e-02,  1.42048144e-02,
                        -9.70116545e-02,  2.19710145e-02, -1.74431868e-02,
                        9.77980681e-02, -3.07681786e-02,  4.86547051e-02,
                        3.46522702e-02],
                        [-3.55657771e-02,  1.27490926e-02, -6.44541416e-02,
                        6.21979023e-02, -6.01156064e-02, -3.29055578e-02,
                        -7.43812334e-02,  9.92552741e-02, -4.15109458e-02,
                        -1.68628936e-02],
                        [-2.83672852e-02, -9.34593242e-03, -2.38763890e-02,
                        2.46906306e-02,  4.77244924e-02,  3.08378967e-02,
                        2.42155668e-02, -4.88641608e-02,  4.67557801e-02,
                        2.70300837e-02],
                        [ 1.09477876e-02,  3.62709518e-02, -3.96666167e-02,
                        -1.33719190e-02, -8.42103519e-02,  5.13786965e-02,
                        3.19077231e-02,  4.23041015e-02,  4.95201320e-03,
                        3.35571515e-02]],

                    [[ 8.87169639e-02,  6.47219835e-02,  4.97651485e-02,
                        -3.62152997e-02,  3.54194701e-02, -6.04386307e-02,
                        9.84582943e-02, -3.57199573e-02, -1.81256738e-02,
                        3.39344011e-02],
                        [ 3.83090884e-02,  7.95874917e-02, -1.44143009e-02,
                        5.66571582e-02,  9.24080814e-04,  8.03379707e-03,
                        -1.09337474e-04,  2.43805848e-02,  1.54228581e-02,
                        2.53311817e-04],
                        [ 1.88393219e-02, -2.78652907e-02,  5.83875066e-02,
                        -1.94994621e-02,  5.63537316e-02, -2.64069612e-03,
                        -9.67920494e-03, -5.71776358e-02, -3.63317913e-02,
                        1.74743092e-03],
                        [-2.88134964e-02, -2.67669291e-02, -7.56609909e-02,
                        1.86957633e-01,  6.34014664e-03,  2.59981605e-02,
                        -1.30324352e-01,  4.43228910e-02,  1.90267860e-02,
                        -4.35082085e-02],
                        [-5.83086629e-05, -1.77846788e-02,  3.20122031e-02,
                        -6.89191921e-03,  8.89139707e-02, -1.65856021e-02,
                        4.91265252e-02, -2.52047026e-02,  2.09365122e-02,
                        -3.37996316e-02],
                        [ 2.55052526e-02, -2.34285334e-02,  8.31340689e-03,
                        2.51638334e-02, -1.73283825e-02,  8.94737527e-02,
                        9.15145225e-04, -2.78385482e-02,  5.03134437e-02,
                        2.14411050e-02],
                        [ 1.35646738e-01,  2.53860817e-02,  5.19412087e-02,
                        -8.66241778e-02,  2.57536672e-02, -5.45035452e-02,
                        1.32186760e-01, -5.08853021e-02,  2.90061348e-02,
                        2.38078409e-02],
                        [ 2.40021338e-02, -1.22914562e-02, -4.18986177e-02,
                        5.26232850e-02, -2.29507203e-02, -9.50748947e-03,
                        -1.27111240e-01,  8.95533375e-02,  9.95697230e-03,
                        1.22545694e-02],
                        [-6.48514187e-03, -4.59764166e-02, -3.53452652e-02,
                        5.03008393e-03, -1.51131365e-03,  3.72577124e-02,
                        2.08455351e-02, -7.39595170e-02,  3.16176636e-02,
                        -6.76054320e-02],
                        [ 2.04486209e-02, -2.17595887e-02,  1.66759481e-02,
                        3.58136479e-03, -4.72908792e-02,  2.20870528e-02,
                        3.88671169e-02, -5.18474596e-02,  2.42848519e-02,
                        6.65222224e-02]]]),
                "theta_matrices": np.array([
                    [[-1.24258272e-02, -2.48792939e-03, -1.33725432e-03,
                        4.94858554e-03, -2.45746258e-03,  4.13903853e-03,
                        -8.45205786e-03,  1.82378813e-03,  4.58818891e-03,
                        -4.39765095e-03],
                        [ 3.07648447e-04, -1.21400134e-02,  5.15827900e-03,
                        -1.08821831e-02,  5.93097513e-03,  5.27533952e-04,
                        6.33023372e-03, -1.29467921e-02,  8.58845621e-03,
                        2.82132333e-03],
                        [-1.87691531e-02, -1.48474786e-03, -4.95524673e-03,
                        1.45196854e-02, -1.49178974e-02, -3.22808195e-03,
                        -1.53246766e-02,  9.31192281e-03, -2.36650763e-03,
                        -2.41818900e-03],
                        [ 5.56227658e-03, -1.43666330e-02,  1.40918841e-03,
                        -2.33318522e-02,  3.79824729e-03, -4.60253594e-03,
                        1.38736777e-02, -8.44284141e-03,  1.02764506e-02,
                        2.12563209e-04],
                        [-1.20320489e-02,  2.34933688e-03, -3.88344049e-03,
                        4.04891931e-03, -9.18144293e-03,  5.71063408e-03,
                        -8.16866460e-03,  5.85978455e-03, -1.59987715e-03,
                        2.12315448e-03],
                        [ 1.30040653e-02, -5.30343875e-03,  4.57471455e-03,
                        -1.23179319e-02,  1.29708591e-02, -6.71973785e-03,
                        8.95664294e-03, -3.73216318e-03,  5.54622637e-03,
                        -4.91403473e-03],
                        [-2.25950197e-02,  9.38192273e-03, -6.18060951e-03,
                        3.25576964e-02, -2.02771350e-02, -1.80421646e-03,
                        -2.93889695e-02,  1.44161539e-02, -7.20228836e-03,
                        -4.34714718e-03],
                        [ 8.30295753e-03, -1.29201886e-03,  2.98329074e-03,
                        -8.76808265e-03,  1.40486958e-02,  7.40898417e-03,
                        1.26687360e-02, -1.10536110e-02,  1.40725392e-02,
                        3.14433817e-03],
                        [ 3.40907531e-03, -3.31354073e-03,  7.55579613e-04,
                        -6.56505910e-03, -3.74595753e-03, -1.15764055e-02,
                        3.96573803e-04,  3.80107978e-03, -7.47282837e-03,
                        -3.95049635e-03],
                        [-1.79037844e-03, -2.19249520e-03,  1.69863473e-03,
                        -1.84603556e-03, -3.43265439e-03, -5.75255647e-03,
                        -1.75742310e-03, -8.20702433e-04, -1.48130975e-03,
                        -3.88986195e-03]],

                    [[-1.21008006e-02, -7.24972402e-03, -5.78670180e-03,
                        6.40586362e-03, -1.34716586e-03,  6.04219991e-03,
                        -7.72328257e-03,  3.30721437e-03,  4.63999691e-03,
                        3.90474068e-03],
                        [ 4.42433243e-03, -2.37400490e-03,  5.23563656e-03,
                        -2.16671023e-02,  1.38683161e-02,  9.04635855e-03,
                        1.74385216e-02, -1.96317271e-02,  1.31289108e-02,
                        -2.05542305e-03],
                        [-1.83606571e-02,  6.99792570e-03, -8.07415575e-03,
                        2.64940849e-02, -9.07742305e-03,  9.97182551e-05,
                        -2.72607110e-02,  1.49577844e-02, -4.20265812e-03,
                        8.35848067e-04],
                        [ 1.58558764e-02, -1.38487650e-02,  1.27422936e-03,
                        -4.44426312e-02,  9.21583811e-03, -1.20203921e-02,
                        2.46805291e-02, -1.27223962e-02,  1.45396440e-03,
                        4.07736460e-03],
                        [-3.65223825e-03, -8.93248071e-03, -5.21151216e-04,
                        2.51746092e-03, -1.41198291e-02,  4.67573574e-03,
                        -8.68782390e-03,  8.73538074e-03, -5.52251217e-03,
                        1.27607441e-02],
                        [ 6.16841233e-03, -1.76615405e-03,  2.38724070e-03,
                        -2.40630077e-02,  2.08686038e-02, -9.54843065e-03,
                        9.56694572e-03, -1.32288633e-02,  3.64774079e-03,
                        -1.47064970e-02],
                        [-4.55117024e-02,  1.81684267e-02, -2.19417007e-02,
                        5.93078155e-02, -2.58522097e-02,  1.19264270e-02,
                        -5.30177565e-02,  4.59369695e-02, -5.12240819e-03,
                        1.21576236e-03],
                        [ 8.19009426e-03, -2.63071957e-03,  1.25204133e-02,
                        -3.52835857e-02,  1.85390849e-02,  7.75340265e-04,
                        2.56488832e-02, -3.38836260e-02,  2.02917312e-02,
                        -3.06908524e-03],
                        [ 2.44844342e-02, -3.82022399e-03,  2.65121310e-03,
                        -8.63921608e-03, -6.24544721e-03, -1.57142050e-02,
                        8.16328137e-03,  1.24137659e-02, -1.80808552e-02,
                        4.13956813e-03],
                        [-7.77573622e-03, -2.38347270e-03, -1.26829827e-03,
                        8.15281795e-04,  4.37239130e-03, -1.03360724e-03,
                        -1.01477676e-02,  7.15733903e-03, -2.96739500e-03,
                        4.17585041e-03]],

                    [[ 2.87063138e-02,  1.47956845e-02,  3.16212161e-02,
                        6.89497250e-03, -1.18780083e-02, -2.13264304e-02,
                        -8.14466130e-04, -8.25683647e-03, -3.20123662e-02,
                        2.70979497e-02],
                        [ 5.80954740e-02,  5.01007350e-02,  1.00498774e-02,
                        9.08170442e-03,  5.32733909e-03,  3.04799679e-02,
                        -2.54669980e-02, -2.59269242e-03,  4.10283038e-03,
                        2.75018636e-02],
                        [ 4.13532136e-02, -6.22666223e-02,  3.82662989e-02,
                        1.94892105e-02,  6.37014943e-02,  1.94298639e-02,
                        6.71784395e-02,  2.31432255e-02,  5.32074962e-04,
                        -3.28836722e-03],
                        [-1.05638548e-02,  4.03286027e-02,  4.63185712e-02,
                        7.48214332e-02,  2.63670941e-02,  1.50320618e-02,
                        -2.81528047e-03, -2.75026306e-02,  6.01112957e-02,
                        6.68578663e-02],
                        [ 1.01017858e-02,  2.08874870e-02,  4.09696937e-02,
                        -3.04117475e-03,  1.98415069e-02, -7.95825152e-02,
                        -4.20926574e-03, -1.79079445e-02, -3.32240983e-03,
                        -1.58050227e-02],
                        [-3.34392310e-02,  2.37921634e-02,  3.82975912e-03,
                        5.83465821e-02, -5.58346603e-02,  4.34100926e-02,
                        4.19711863e-03, -1.40733443e-03,  2.44317700e-02,
                        1.98514337e-02],
                        [ 1.89723212e-02, -4.18051243e-02,  4.00062741e-02,
                        -1.83076080e-02,  1.36845238e-02,  1.17051756e-02,
                        2.54091313e-02,  4.98087828e-02,  4.87471114e-02,
                        -2.91822553e-02],
                        [-1.04580957e-03,  5.56751216e-03,  4.13673991e-02,
                        5.04584947e-03, -2.35097894e-02,  2.09858471e-03,
                        -9.18768934e-03,  5.51526494e-02, -8.03822086e-02,
                        4.25856821e-02],
                        [ 1.80040447e-02, -5.61493597e-03, -4.66254183e-02,
                        -2.76833598e-02,  5.14059577e-02,  1.79793574e-03,
                        2.07881348e-02, -2.72597442e-02,  6.00571372e-02,
                        6.48132485e-04],
                        [-3.03990939e-02, -4.06423100e-02,  1.92382195e-02,
                        4.25147671e-02, -1.87961822e-02,  8.24264527e-03,
                        3.66011394e-02, -3.06681302e-02,  7.58775531e-02,
                        5.89010666e-05]]]),
                "sigma": np.array([
                [ 0.4968,  0.0111,  0.249 ,  0.0005, -0.0074,  0.254 , -0.0132,
                    0.0054,  0.0005,  0.2635],
                [ 0.0111,  0.498 ,  0.0005, -0.0074,  0.254 , -0.0132,  0.0054,
                    0.0005,  0.2635, -0.016 ],
                [ 0.249 ,  0.0005, -0.0148,  0.254 , -0.0132,  0.0054,  0.0005,
                    0.2635, -0.016 , -0.0115],
                [ 0.0005, -0.0074,  0.254 , -0.0264,  0.0054,  0.0005,  0.2635,
                    -0.016 , -0.0115,  0.0142],
                [-0.0074,  0.254 , -0.0132,  0.0054,  0.05 ,  0.2635, -0.016 ,
                    -0.0115,  0.0142,  0.0068],
                [ 0.254 , -0.0132,  0.0054,  0.0005,  0.2635, -0.032 , -0.0115,
                    0.0142,  0.0068,  0.2479],
                [-0.0132,  0.0054,  0.0005,  0.2635, -0.016 , -0.0115,  0.0284,
                    0.0068,  0.2479,  0.0051],
                [ 0.0054,  0.0005,  0.2635, -0.016 , -0.0115,  0.0142,  0.0068,
                    0.4958,  0.0051, -0.0053],
                [ 0.0005,  0.2635, -0.016 , -0.0115,  0.0142,  0.0068,  0.2479,
                    0.0051, -0.0106, -0.056],
                [ 0.2635, -0.016 , -0.0115,  0.0142,  0.0068,  0.2479,  0.0051,
                    -0.0053, -0.056,  0.0006]]),
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [5],
                "signals": [self.exog]
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 5],
                "signals": [self.interv_sig, self.exog]
            },
            initial_condition=np.array([
                [-2.81111324,  0.22267988, -2.07542116,  0.53394164, -0.35001051,
                0.05721839,  0.83864668, -2.97425894, -1.42596918,  1.25991323],
                [-2.86369369,  0.03248342,  1.90423438,  1.8642061 ,  1.40331635,
                -2.32994765, -2.78480841,  0.58921383, -1.60684836,  2.44905434]]),
            start_time=0,
            timestep=1,
            rng = np.random.default_rng(SEED)
        )


# Wilson Cowan Data Generators.
WC_MAX_NUM_STEPS = 11_000
WC_TIMESTEP = 0.02
WC_AMAX = 0.6
WC_AMIN = 0.3
WC_FMAX = 3.0
WC_FMIN = 2
WC_MAX_T = WC_TIMESTEP * WC_MAX_NUM_STEPS

WC_SLOPE = 2
WC_THRESH = 0.5

WC_MULTI_CONF_INTERV_CONST = 0.55

class WilsonCowanMultiConf(DataGenerator):

    target_idx = 1


    interv_sig = lambda self, t: WC_MULTI_CONF_INTERV_CONST
    exog = randsig(
        max_T=WC_MAX_T, amax=WC_AMAX, amin=WC_AMIN, fmax=WC_FMAX, fmin=WC_FMIN,
        rng=np.random.default_rng(123)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.WilsonCowan,
            model_params={
                "tau": WC_SLOPE,
                "mu": WC_THRESH,
                "adjacency_matrix": np.array([
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]),
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2],
                "signals": [self.exog],
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2],
                "signals": [self.interv_sig, self.exog],
            },
            initial_condition=np.array([[0.15366136, 0.1693033 , 0.50596431, 0.65811887]]),
            timestep=WC_TIMESTEP,
        )


WC_DOWNSTREAM_INTERV_CONST = 0.55

class WilsonCowanDownstream(DataGenerator):

    target_idx = 1

    interv = lambda self, t: WC_DOWNSTREAM_INTERV_CONST
    exog = randsig(
        max_T=WC_MAX_T, amax=WC_AMAX, amin=WC_AMIN, fmax=WC_FMAX, fmin=WC_FMIN,
        rng=np.random.default_rng(124)
    )

    def __init__(self):
        super().__init__(
            model_type=interfere.dynamics.WilsonCowan,
            model_params={
                "tau": WC_SLOPE,
                "mu": WC_THRESH,
                "adjacency_matrix": np.array([
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]),
            },
            obs_intervention_type=interfere.SignalIntervention,
            obs_intervention_params={
                "intervened_idxs": [2],
                "signals": [self.exog],
            },
            do_intervention_type=interfere.SignalIntervention,
            do_intervention_params={
                "intervened_idxs": [0, 2],
                "signals": [self.interv, self.exog],
            },
            initial_condition=np.array([[0.76758088, 0.10922746, 0.79759653, 0.96874591]]),
            timestep=WC_TIMESTEP,
        )


ALL_MODELS = [
    ArithmeticBrownianMotion,
    AttractingFixedPoint4D,
    Belozyorov1,
    Belozyorov2,
    Belozyorov3,
    CoupledLogisticMapAllToAll,
    CoupledLogisticMapTwoCycles,
    CoupledLogisticMapOneCycle,
    CoupledMapLatticeChaoticBrownian,
    CoupledMapLatticeChaoticTravelingWave,
    CoupledMapLatticeDefectTurbulence,
    CoupledMapLatticePatternSelection,
    CoupledMapLatticeSpatioTempChaos,
    CoupledMapLatticeSpatioTempIntermit,
    CoupledMapLatticeTravelingWave,
    DampedOscillator1,
    DampedOscillator2,
    GeometricBrownianMotion1,
    GeometricBrownianMotion2,
    HodgkinHuxley1,
    HodgkinHuxley2Chain,
    HodgkinHuxley3Grid,
    ImaginaryRoots4D,
    KuramotoOscilator1,
    KuramotoOscilator2,
    KuramotoOscilator3,
    KuramotoOscilator4,
    KuramotoSakaguchi1,
    KuramotoSakaguchi2,
    KuramotoSakaguchi3,
    Liping3DQuadFinance1,
    Liping3DQuadFinance2,
    LotkaVoltera1,
    LotkaVoltera2,
    LotkaVoltera3,
    LotkaVoltera4,
    MichaelisMenten1,
    MichaelisMenten2,
    MichaelisMenten3,
    MooreSpiegel1,
    MooreSpiegel2,
    MooreSpiegel3,
    MutualisticPop1,
    MutualisticPop1,
    MutualisticPop1,
    OrnsteinUhlenbeck1,
    OrnsteinUhlenbeck2,
    OrnsteinUhlenbeck3,
    PlantedTank1,
    PlantedTank2,
    PlantedTank3,
    VARMA1SpatiotempChaos,
    VARMA2ChaoticBrownian,
    VARMA3LotkaVoltera, 
    LotkaVolteraConfound,
    LotkaVolteraMediator,
    CoupledLogisticMapConfound,
    CoupledLogisticMapBlockableConfound,
    WilsonCowanDownstream,
    WilsonCowanMultiConf,
    Lorenz1,
    Lorenz2,
    Rossler1,
    Rossler2,
    Thomas1,
    Thomas2
]