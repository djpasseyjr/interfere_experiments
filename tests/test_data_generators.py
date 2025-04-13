from typing import Type

import pytest
import numpy as np
import interfere
import interfere_experiments as ie


@pytest.mark.parametrize("data_gen_type", ie.data_generators.ALL_MODELS)
def test_generate(
    data_gen_type: Type[ie.data_generators.DataGenerator]
):
    dg = data_gen_type()
    data = dg.generate_data(50, 50)
    assert data.train_states.shape == (50, dg.initial_condition.shape[1]), (
        f"Incorrect train shape generated for {type(dg).__name__} ."
        f"\n\tExpected: {(50, dg.initial_condition.shape[1])} "
        f"\n\tActual: {data.train_states.shape}"
    )
    assert data.forecast_states.shape == (
        50, dg.initial_condition.shape[1]), (
            f"Incorrect forecast shape generated for {type(dg).__name__}"
            f"\n\tExpected: {(50, dg.initial_condition.shape[1])} "
            f"\n\tActual: {data.forecast_states.shape}"
    )
    assert data.interv_states.shape == (50, dg.initial_condition.shape[1]), (
        f"Incorrect interv shape generated for {type(dg).__name__}"
            f"\n\tExpected: {(50, dg.initial_condition.shape[1])} "
            f"\n\tActual: {data.interv_states.shape}"
    )

    data = dg.generate_data(50, 50, num_burn_in_states=25)
    assert data.train_states.shape == (50, dg.initial_condition.shape[1]), (
        f"Incorrect train shape generated for {type(dg).__name__} with burn in."
            f"\n\tExpected: {(50, dg.initial_condition.shape[1])} "
            f"\n\tActual: {data.train_states.shape}"
    )
    assert data.forecast_states.shape == (
        50, dg.initial_condition.shape[1]), (
            f"Incorrect forecast shape generated for {type(dg).__name__} with burn in."
            f"\n\tExpected: {(50, dg.initial_condition.shape[1])} "
            f"\n\tActual: {data.forecast_states.shape}"
    )
    assert data.interv_states.shape == (50, dg.initial_condition.shape[1]), (
        f"Incorrect interv shape generated for {type(dg).__name__} with burn in."
            f"\n\tExpected: {(50, dg.initial_condition.shape[1])} "
            f"\n\tActual: {data.interv_states.shape}"
    )


def test_burn_in():
    dg = ie.data_generators.DampedOscillator2()
    dg.model_params["sigma"] = 0
    dg.model_params["measurement_noise_std"] = np.zeros(2)
    burn = dg.generate_data(num_burn_in_states=50)

    dg.rng = np.random.default_rng(ie.data_generators.SEED)
    no_burn = dg.generate_data()

    # I got stuck in indexing errors for literally hours trying to test this 
    # but then I plotted it and saw it was working so brute force it is :)
    offset_diffs = [
        np.sum(no_burn.train_states[i:, :]  - burn.train_states[:-i])
        for i in range(1, no_burn.train_states.shape[0])
    ]
    assert np.any(offset_diffs)

    i_star = np.argmin(np.abs(offset_diffs))

    assert np.all(no_burn.train_t[i_star + 1:] == burn.train_t[:-(i_star + 1)])


def test_random_sig():

    # Check that amplitude works as expected.
    sig1 = ie.data_generators.randsig(10, amax=1)
    sig2 = ie.data_generators.randsig(10, amax=3)

    t = np.linspace(0, 10, 1000)

    assert np.mean(sig1(t)) < np.mean(sig2(t))

    # Check that frequency works as expected.
    sig1 = ie.data_generators.randsig(10, fmax=10)
    sig2 = ie.data_generators.randsig(10, fmax=3)

    assert np.std(sig1(t)) > np.std(sig2(t))

# Test on only the first ten generators to save time.
@pytest.mark.parametrize("dg_type", [
    ie.data_generators.AttractingFixedPoint4D,
])
def test_generate_data_and_downsample(
        dg_type: type[ie.data_generators.DataGenerator]
    ):
    """Tests that generate_data_and_downsample produces the correct data.
    """
    new_timestep = 0.00001
    dg = dg_type()
    dg.model_params["sigma"] = 0.0

    data1 = dg.generate_data(
        num_train_obs = 10,
        num_forecast_obs = 5,
        num_burn_in_states = 5,
    )


    # If the model is discrete time, assert error is raised.
    if isinstance(
        dg.model_type, interfere.dynamics.base.DiscreteTimeDynamics):
        with pytest.raises(NotImplementedError):
            dg.generate_data_and_downsample(
                new_timestep = new_timestep,
                num_train_obs = 10,
                num_forecast_obs = 5,
                num_burn_in_states = 5,
            ) 
    else:
        # Test the downsample method.
        data2 = dg.generate_data_and_downsample(
            new_timestep = new_timestep,
            num_train_obs = 10,
            num_forecast_obs = 5,
            num_burn_in_states = 5,
        )        

        assert np.allclose(
            data1.train_prior_t, data2.train_prior_t, atol=0.1, rtol=0)
        assert np.allclose(
            data1.train_prior_states, data2.train_prior_states, atol=0.1, rtol=0
        )
        assert np.allclose(data1.train_t, data2.train_t, atol=0.1, rtol=0)
        assert np.allclose(
            data1.train_states, data2.train_states, atol=0.1, rtol=0)
        assert np.allclose(data1.forecast_t, data2.forecast_t, atol=0.1, rtol=0)
        assert np.allclose(
            data1.forecast_states, data2.forecast_states, atol=0.1, rtol=0)
        assert np.allclose(
            data1.interv_states, data2.interv_states, atol=0.1, rtol=0)
        