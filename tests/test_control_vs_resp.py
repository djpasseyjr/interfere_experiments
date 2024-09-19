from typing import Any, Dict
import interfere
from interfere.methods.base import BaseInferenceMethod
import interfere_experiments as ie

import numpy as np
import pytest

SEED = 11
RNG = np.random.default_rng(SEED)

GEN_DATA_ARGS = [
    {
        "model": ie.quick_models.gut_check_belozyorov(),
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

CTRL_V_RESP_DATA = [
        ie.control_vs_resp.generate_data(**kwargs)
        for kwargs in GEN_DATA_ARGS
]

METHODS = [interfere.methods.VAR, interfere.methods.ResComp]


def test_control_vs_resp_data_shape():
    """Tests that the ControlVsRespData class raises ValueErrors correctly."""
    n_train_prior_obs = 5
    n_train_obs = 10
    n_forecast_obs = 5
    n_cols = 3
    train_prior_t = np.arange(n_train_prior_obs)
    train_prior_states = np.random.random((n_train_prior_obs, n_cols))
    train_t = np.arange(n_train_obs)
    train_states = np.random.random((n_train_obs, n_cols))
    forecast_t = np.arange(n_forecast_obs)
    forecast_states = np.random.random((n_forecast_obs, n_cols))
    interv_states = np.random.random((n_forecast_obs, n_cols))
    intervention = interfere.PerfectIntervention(0, 0)

    # Bad train prior data.
    with pytest.raises(ValueError, match="train_prior_t"):
        bad_train_prior_t = np.arange(n_train_obs)
        ie.control_vs_resp.ControlVsRespData(
            bad_train_prior_t, train_prior_states, train_t, train_states,
            forecast_t, forecast_states, intervention, interv_states
        )

    # Bad train data.
    with pytest.raises(ValueError, match="train_t"):
        bad_train_t = np.arange(n_forecast_obs)
        ie.control_vs_resp.ControlVsRespData(
            train_prior_t, train_prior_states, bad_train_t, train_states,
            forecast_t, forecast_states, intervention, interv_states
        )

    # Bad forecast data.
    with pytest.raises(ValueError, match="forecast_t"):
        bad_forecast_t = np.arange(n_train_obs)
        ie.control_vs_resp.ControlVsRespData(
            train_prior_t, train_prior_states, train_t, train_states,
            bad_forecast_t, forecast_states, intervention, interv_states
        )

    # Bad intervention data.
    with pytest.raises(ValueError, match="interv_states"):
        bad_interv_states = np.random.random((n_train_obs, n_cols))
        ie.control_vs_resp.ControlVsRespData(
            train_prior_t, train_prior_states, train_t, train_states,
            forecast_t, forecast_states, intervention, bad_interv_states
        )

    # Check that bad state array shape raise a ValueError.
    with pytest.raises(ValueError, match="number of columns"):
        bad_train_states = np.random.random((n_train_obs, n_cols + 1))
        ie.control_vs_resp.ControlVsRespData(
            train_prior_t, train_prior_states, train_t, bad_train_states,
            forecast_t, forecast_states, intervention, interv_states
        )


@pytest.mark.parametrize(
    "cvr_args, cvr_data", zip(GEN_DATA_ARGS, CTRL_V_RESP_DATA))
class TestDataGen:
    """Tests that ie.control_vs_resp.generate_data works correctly."""
    

    def test_timestep(
        self,
        cvr_args: Dict[str, Any],
        cvr_data: ie.control_vs_resp.ControlVsRespData
    ):
        """Tests that the timestep of all time arrays is correct.
        
        Args:
            cvr_args: A dictionary of args passed to    
                ie.control_vs_resp.generate_data().
            cvr_data: The return value of 
                ie.control_vs_resp.generate_data()
                corresponding to the args.
        """

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


    def test_state_array_shape(
        self,
        cvr_args: Dict[str, Any],
        cvr_data: ie.control_vs_resp.ControlVsRespData
    ):
        """Tests that the shape of all state arrays is correct.

        Args:
            cvr_args: A dictionary of args passed to    
                ie.control_vs_resp.generate_data().
            cvr_data: The return value of 
                ie.control_vs_resp.generate_data()
                corresponding to the args.
    """
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


@pytest.mark.parametrize("cvr_data", CTRL_V_RESP_DATA)
@pytest.mark.parametrize("method_type", METHODS)
def test_predict_array_shapes(
    cvr_data: ie.control_vs_resp.ControlVsRespData,
    method_type: BaseInferenceMethod
):
    """Tests that the shape of all predicted arrays is correct.

    Args:
        cvr_data: The return value of 
            ie.control_vs_resp.generate_data()
            corresponding to the args.
    """
    tr_pred, fc_pred, ivn_pred = ie.control_vs_resp.make_predictions(
        method_type(**method_type.get_test_params()),
        cvr_data.train_prior_t,
        cvr_data.train_prior_states,
        cvr_data.train_t,
        cvr_data.train_states,
        cvr_data.forecast_t,
        cvr_data.intervention
    )

    assert tr_pred.shape == cvr_data.train_states.shape, (
        "The array tr_pred is the incorrect shape: "
        f"\nArray shape: {tr_pred.shape} != "
        f"Expected shape: {cvr_data.train_states.shape}"
    )

    assert fc_pred.shape == cvr_data.forecast_states.shape, (
        "The array fc_pred is the incorrect shape: "
        f"\nArray shape: {fc_pred.shape} != "
        f"Expected shape: {cvr_data.forecast_states.shape}"
    )

    assert ivn_pred.shape == cvr_data.interv_states.shape, (
        "The array ivn_pred is the incorrect shape: "
        f"\nArray shape: {ivn_pred.shape} != "
        f"Expected shape: {cvr_data.interv_states.shape}"
    )


def test_visualize():
    """Tests that ie.control_vs_resp.visualize works correctly."""

    method_type = interfere.methods.VAR
    model = ie.quick_models.gut_check_belozyorov()
    cvr_data = CTRL_V_RESP_DATA[0]
    tr_pred, fc_pred, ivn_pred = ie.control_vs_resp.make_predictions(
        method_type(**method_type.get_test_params()),
        cvr_data.train_prior_t,
        cvr_data.train_prior_states,
        cvr_data.train_t,
        cvr_data.train_states,
        cvr_data.forecast_t,
        cvr_data.intervention
    )
    img = ie.control_vs_resp.visualize(
        model, method_type, cvr_data, tr_pred, fc_pred, ivn_pred
    )

    assert img.size, (
        "The function ie.control_vs_resp.visualize() produced an empty image."
    )

    # Check that the image is not grayscale.
    img_array = np.array(img)
    assert not np.all(img_array[:, :, :1] == img_array), (
        "ie.control_vs_resp.visualize() produced an grayscale image. "
        f"img.getextrema() = {img.getextrema()}"
    )