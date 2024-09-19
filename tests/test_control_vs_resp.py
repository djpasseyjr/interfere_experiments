from typing import Any, Dict, Iterable
from unittest.mock import Mock, MagicMock

import interfere
from interfere.methods.base import BaseInferenceMethod
import interfere_experiments as ie
import numpy as np
import PIL
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
        "rng": np.random.default_rng(SEED),
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
    {
        "model": ie.quick_models.gut_check_coupled_logistic(),
        "num_train_obs": 10,
        "num_forecast_obs": 5,
        "timestep": 1,
        "intervention": interfere.PerfectIntervention(2, 0.2),
        "rng": RNG,
        "train_prior_states": RNG.random((3, 10)),
        "lags": 10,
    }
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
    "gen_data_args, cvr_data", zip(GEN_DATA_ARGS, CTRL_V_RESP_DATA))
class TestDataGen:
    """Tests that ie.control_vs_resp.generate_data works correctly."""
    

    def test_timestep(
        self,
        gen_data_args: Dict[str, Any],
        cvr_data: ie.control_vs_resp.ControlVsRespData
    ):
        """Tests that the timestep of all time arrays is correct.
        
        Args:
            gen_data_args: A dictionary of args passed to    
                ie.control_vs_resp.generate_data().
            cvr_data: The return value of 
                ie.control_vs_resp.generate_data()
                corresponding to the args.
        """

        dt = gen_data_args["timestep"]
        for time_array in [
            cvr_data.train_prior_t, cvr_data.train_t, cvr_data.forecast_t
        ]:
            assert np.allclose(np.diff(time_array), dt), (
                "Timesteps do not match passed timestep for "
                f"{type(gen_data_args['model'])}"
                f"\nForecast time steps: {np.diff(time_array)}"
                f"\nTime step arg: {dt}"
            )


    def test_state_array_shape(
        self,
        gen_data_args: Dict[str, Any],
        cvr_data: ie.control_vs_resp.ControlVsRespData
    ):
        """Tests that the shape of all state arrays is correct.

        Args:
            gen_data_args (Dict[str, Any]): A dictionary of args passed to    
                ie.control_vs_resp.generate_data().
            cvr_data: The return value of 
                ie.control_vs_resp.generate_data()
                corresponding to the args.
    """
        dim = gen_data_args["model"].dim
        lags = gen_data_args["lags"]
        prior_states = gen_data_args["train_prior_states"]
        nprior_obs = lags if prior_states is None else prior_states.shape[0]
        num_train_obs = gen_data_args["num_train_obs"]
        num_forecast_obs = gen_data_args["num_forecast_obs"]

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


@pytest.mark.parametrize("gen_data_args", GEN_DATA_ARGS)
def test_generate_data_rng_is_reproducible(
    gen_data_args: Dict[str, Any],
):
    """Tests that rng argument leads to reproducible noise.

    Args:
        gen_data_args (Dict[str, Any]): A dictionary of args passed to    
            ie.control_vs_resp.generate_data().
        cvr_data: The return value of 
            ie.control_vs_resp.generate_data()
            corresponding to the args.
    """
    rng = np.random.default_rng(SEED)
    data = ie.control_vs_resp.generate_data(**{
        **gen_data_args,
        "rng": rng
    })

    rng = np.random.default_rng(SEED)
    rerun_data = ie.control_vs_resp.generate_data(**{
        **gen_data_args,
        "rng": rng
    })

    assert rerun_data == data, (
        "Range does not preserve randomness for "
        "ie.control_vs_respons.generate_data() on "
        f"{type(gen_data_args['model'])}"
    )


@pytest.mark.parametrize("cvr_data", CTRL_V_RESP_DATA)
@pytest.mark.parametrize("method_type", METHODS)
def test_predict_array_shapes(
    cvr_data: ie.control_vs_resp.ControlVsRespData,
    method_type: BaseInferenceMethod
):
    """Tests that the shape of all predicted arrays is correct.

    Args:
        cvr_data (Dict[str, Any]): The return value of 
            ie.control_vs_resp.generate_data()
            corresponding to the args.
        method_type (BaseInferenceMethod): The method to use to generate
            forecasts.
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


@pytest.mark.parametrize("gen_data_args", GEN_DATA_ARGS)
def test_optuna_obj_data(
    gen_data_args: Dict[str, Any],
):
    """Tests that CVROptunaObjective class creates time series correctly.
    
    Args:
        gen_data_args (Dict[str, Any]): A dictionary of args passed to    
            ie.control_vs_resp.generate_data().
    """
    objective = ie.control_vs_resp.CVROptunaObjective(
        method_type = interfere.methods.VAR,
        metrics = (interfere.metrics.RootMeanStandardizedSquaredError(),),
        metric_directions = ("minimize",),
        **gen_data_args
    )

    assert isinstance(objective.data, ie.control_vs_resp.ControlVsRespData), (
        "Optuan objective class did not create ControlVsRespData."
    )

    dim = gen_data_args["model"].dim
    lags = gen_data_args["lags"]
    prior_states = gen_data_args["train_prior_states"]
    nprior_obs = lags if prior_states is None else prior_states.shape[0]
    num_train_obs = gen_data_args["num_train_obs"]
    num_forecast_obs = gen_data_args["num_forecast_obs"]

    assert objective.data.train_prior_t.shape == (nprior_obs,), (
        "The array train_prior_t is the incorrect shape: "
        f"\nArray shape: {objective.data.train_prior_t.shape} != "
        f"Expected shape: {(lags,)}"
    )
    assert objective.data.train_prior_states.shape == (nprior_obs, dim), (
        "The array train_prior_states is the incorrect shape: "
        f"\nArray shape: {objective.data.train_prior_states.shape} != "
        f"Expected shape: {(lags, dim)}"
    )

    assert objective.data.train_t.shape == (num_train_obs,), (
        "The array train_t is the incorrect shape: "
        f"\nArray shape: {objective.data.train_t.shape} != "
        f"Expected shape: {(num_train_obs,)}"
    )

    assert objective.data.train_states.shape == (num_train_obs, dim), (
        "The array train_states is the incorrect shape: "
        f"\nArray shape: {objective.data.train_states.shape} != "
        f"Expected shape: {(num_train_obs, dim)}"
    )

    assert objective.data.forecast_t.shape == (num_forecast_obs,), (
        "The array forecast_t is the incorrect shape: "
        f"\nArray shape: {objective.data.forecast_t.shape} != "
        f"Expected shape: {(num_forecast_obs, dim)}"
    )

    assert objective.data.forecast_states.shape == (num_forecast_obs, dim), (
        "The array forecast_states is the incorrect shape: "
        f"\nArray shape: {objective.data.forecast_states.shape} != "
        f"Expected shape: {(num_forecast_obs, dim)}"
    )

    assert objective.data.interv_states.shape == (num_forecast_obs, dim), (
        "The array interv_states is the incorrect shape: "
        f"\nArray shape: {objective.data.interv_states.shape} != "
        f"Expected shape: {(num_forecast_obs, dim)}"
    )


@pytest.mark.parametrize("metrics, metric_directions", [
    (
        [
            interfere.metrics.RootMeanStandardizedSquaredError()
        ], 
        ["minimize"]
    ),
    (
        [
           interfere.metrics.RootMeanStandardizedSquaredError(), 
            interfere.metrics.TTestDirectionalChangeAccuracy()
        ],
        ["minimize", "maximize"]
    ),
])
def test_optuna_obj_metrics(
    metrics: Iterable[interfere.metrics.CounterfactualForecastingMetric],
    metric_directions: Iterable[str]
):
    """Tests that CVROptunaObjective handles metrics correctly.

    Args:
        metrics (Iterable[CounterfactualForecastingMetric]): A collection of
            metrics for measuring success at the counterfactual forecasting
            problem.
        metric_directions (Iterable[str]): Must only contain "maximize" or
            "minimize". To be passed to the optuna study. 
    """
    gen_data_args = GEN_DATA_ARGS[0]

    # Initialize an objective.
    objective = ie.control_vs_resp.CVROptunaObjective(
        **{
            **gen_data_args, 
            "num_train_obs": 20,
            "num_forecast_obs": 50,
        },
        method_type = Mock(),
        metrics = metrics,
        metric_directions = metric_directions,
        hyperparam_func=MagicMock(return_value={}),
    )

    # Create mock methods that predict different data.
    mock_methods = [Mock() for i in range(3)]

    for mm, X in zip(
        mock_methods,
        [
            objective.data.train_states,
            objective.data.forecast_states,
            objective.data.interv_states
        ]
    ):
        # Create methods that predict true training data only, true forecast
        # data only and true intervention data only.
        mm.predict = (
            lambda t, prior_endog_states=None, X=X.copy(), **kwargs:
            (
                X[:, :prior_endog_states.shape[1]] 
                if len(t) == X.shape[0] 
                else np.random.rand(len(t), prior_endog_states.shape[1])
            )
        )

    # Check that mock methods are working correctly.
    mock_train = mock_methods[0].predict(
        objective.data.train_t, 
        prior_endog_states = objective.data.train_prior_states
    )

    assert np.all(mock_train == objective.data.train_states), (
        "Mock method does not predict training data correctly."
    )
    
    mock_forecast = mock_methods[1].predict(
        objective.data.forecast_t,
        prior_endog_states = objective.data.train_states
    )

    assert np.all(mock_forecast == objective.data.forecast_states), (
        "Mock method does not predict forecast data correctly."
    )

    mock_interv = mock_methods[2].predict(
        objective.data.forecast_t, 
        prior_endog_states = objective.data.train_states[:, :2]
    )

    assert np.all(mock_interv == objective.data.interv_states[:, :2]), (
        "Mock method does not predict intervention data correctly."
    )


    # Create 
    mock_method_types = [Mock(return_value=mm) for mm in mock_methods]

    series_names = ["train", "forecast", "interv"]
    assert objective.metric_names == [
        sn + "_" + m.name for m in metrics for sn in series_names]
    assert objective.metric_directions == metric_directions * 3

    for mock_method_type, predict_series in zip(
        mock_method_types, series_names
    ):
        mock_method_type.__name__ = "Mock_" + predict_series
        objective.method_type = mock_method_type

        scores = objective(Mock())

        for i, series_name in enumerate(series_names):
            for j, m in enumerate(metrics):
                # Compute score index corresponding to series and metric.
                idx = i * len(metrics) + j

                if series_name == predict_series:
                    # When the series matches the predicted states, all metrics
                    # should have the perfect score.

                    if objective.metric_directions[idx] == "maximize":
                        assert scores[idx] == 1.0, (
                            f"{series_name}_{m.name} scored {scores[idx]}. "
                            "\nExpected score = 1.0"
                            f"\nMocked method predict returned {predict_series}"
                            "_states."
                        )
                    else:
                        assert scores[idx] == 0.0, (
                            f"{series_name}_{m.name} scored {scores[idx]}. "
                            "\nExpected score = 0.0"
                            f"\nMocked method predict returned {predict_series}"
                            "_states."
                        )
                else:
                    # When the current series does not match the predicted
                    # states, no metric should have a perfect score.

                    if objective.metric_directions[idx] == "maximize":
                        assert scores[idx] != 1.0, (
                            f"{series_name}_{m.name} scored {scores[idx]}. "
                            "\nExpected score != 1.0"
                            f"\nMocked method predict returned {predict_series}"
                            "_states."
                        )
                    else:
                        assert scores[idx] != 0.0, (
                            f"{series_name}_{m.name} scored {scores[idx]}. "
                            "\nExpected score != 0.0"
                            f"\nMocked method predict returned {predict_series}"
                            "_states."
                        )


def test_optuna_obj_exception_log():
    """Tests that optuna objective keeps an exception log."""

    # Make a method type that raises value error when it is called after
    # initialization.
    mock_method = Mock()
    mock_method.predict = Mock(
        side_effect=ValueError("Mock mock. Who's there?"))
    mock_method_type = Mock(
        return_value=mock_method
    )
        
    objective = ie.control_vs_resp.CVROptunaObjective(
        **GEN_DATA_ARGS[0], 
        method_type = mock_method_type,
        hyperparam_func=MagicMock(return_value={}),
    )

    mock_trial = Mock()
    idx = 1
    mock_trial.number = idx

    objective(mock_trial)
    assert "Mock mock. Who's there?" in objective.trial_error_log[idx], (
        "Error not logged in objective error logger."
        "\nLog:\n\n"
        f"{objective.trial_error_log[idx]}"
    )

    assert objective.trial_imgs[idx] is None, (
        "Image stored when there was an error."
        f"\nobjective.trial_imgs[idx] = {objective.trial_imgs[idx]}"
    )


def test_optuna_obj_figures():
    """Tests that the CVROptunaObjective stores figures correctly."""

    objective = ie.control_vs_resp.CVROptunaObjective(
        **GEN_DATA_ARGS[0], 
        method_type = interfere.methods.VAR,
        hyperparam_func=MagicMock(return_value={}),
    )

    idx = 88
    mock_trial = Mock()
    mock_trial.number = idx

    objective(mock_trial)

    assert objective.trial_error_log[idx] is "", (
        "Error log not empty."
        f"\nLog: \n\nobjective.trial_error_log[idx]"
        f" = {objective.trial_error_log[idx]}"
    )

    assert isinstance(objective.trial_imgs[idx], PIL.Image.Image), (
        "Image not saved."
        f"\nobjective.trial_imgs[idx] = {objective.trial_imgs[idx]}"
    )