import interfere
import interfere_experiments as ie

import numpy as np

SEED = 11
RNG = np.random.default_rng(SEED)

MAKR_CTRL_V_RESP_ARGS = [
    {
        "model": ie.quick_models.gut_check_belozy(),
        "num_train_obs": 10,
        "num_forecast_obs": 5,
        "timestep": 0.05,
        "intervention": interfere.PerfectIntervention(2, 0.2),
        "rng": RNG,
        "train_prior_states": None,
        "lags": 10,
    },
    {
        "model": ie.quick_models.gut_check_coupled_logistic(),
        "num_train_obs": 10,
        "num_forecast_obs": 5,
        "timestep": 1,
        "intervention": interfere.PerfectIntervention(2, 0.2),
        "rng": RNG,
        "train_prior_states": None,
        "lags": 10,
    },
]

class TestDataGen:

    cvr_data = [
        ie.control_vs_response.make_control_vs_response_data(**args)
        for args in MAKR_CTRL_V_RESP_ARGS
    ]


    def test_timestep(self):
        """Tests that the timestep of all time arrays is correct."""

        for cvr_args, cvr_data in zip(MAKR_CTRL_V_RESP_ARGS, self.cvr_data):

            dt = cvr_args["timestep"]
            for time_array in [
                cvr_data.train_prior_t, cvr_data.train_t, cvr_data.forecast_t
            ]:
                assert np.allclose(np.diff(time_array), dt), (
                    "Timesteps do not match passed timestep for "
                    f"{type(cvr_args['model'])}"
                    f"\nForecast time steps: {np.diff(time_array)}"
                    f"\nTime step arg: {dt}"
                )


    def test_state_array_shape(self):
        """Tests that the shape of all state arrays is correct."""

        for cvr_args, cvr_data in zip(MAKR_CTRL_V_RESP_ARGS, self.cvr_data):

            dim = cvr_args["model"].dim
            lags = cvr_args["lags"]
            prior_states = cvr_args["train_prior_states"]
            nprior_obs = lags if prior_states is None else prior_states.shape[0]
            num_train_obs = cvr_args["num_train_obs"]
            num_forecast_obs = cvr_args["num_forecast_obs"]

            assert cvr_data.train_prior_t.shape == (nprior_obs,), (
                "The array train_prior_t is the incorrect shape: "
                f"\nArray shape: {cvr_data.train_prior_t.shape} != "
                f"Expected shape: {(lags,)}"
            )
            assert cvr_data.train_prior_states.shape == (nprior_obs, dim), (
                "The array train_prior_states is the incorrect shape: "
                f"\nArray shape: {cvr_data.train_prior_states.shape} != "
                f"Expected shape: {(lags, dim)}"
            )

            assert cvr_data.train_t.shape == (num_train_obs,), (
                "The array train_t is the incorrect shape: "
                f"\nArray shape: {cvr_data.train_t.shape} != "
                f"Expected shape: {(num_train_obs,)}"
            )

            assert cvr_data.train_states.shape == (num_train_obs, dim), (
                "The array train_states is the incorrect shape: "
                f"\nArray shape: {cvr_data.train_states.shape} != "
                f"Expected shape: {(num_train_obs, dim)}"
            )

            assert cvr_data.forecast_t.shape == (num_forecast_obs,), (
                "The array forecast_t is the incorrect shape: "
                f"\nArray shape: {cvr_data.forecast_t.shape} != "
                f"Expected shape: {(num_forecast_obs, dim)}"
            )

            assert cvr_data.forecast_states.shape == (num_forecast_obs, dim), (
                "The array forecast_states is the incorrect shape: "
                f"\nArray shape: {cvr_data.forecast_states.shape} != "
                f"Expected shape: {(num_forecast_obs, dim)}"
            )

            assert cvr_data.interv_states.shape == (num_forecast_obs, dim), (
                "The array interv_states is the incorrect shape: "
                f"\nArray shape: {cvr_data.interv_states.shape} != "
                f"Expected shape: {(num_forecast_obs, dim)}"
            )


