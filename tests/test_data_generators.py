from typing import Type

import pytest
import numpy as np

import interfere_experiments as ie


@pytest.mark.parametrize("data_gen_type", ie.data_generators.ALL_MODELS)
def test_generate(
    data_gen_type: Type[ie.data_generators.DataGenerator]
):
    dg = data_gen_type()
    data = dg.generate_data(50, 50)
    assert data.train_states.shape == (50, dg.initial_condition.shape[1]), (
        "Incorrect train shape generated for {model}"
    )
    assert data.forecast_states.shape == (
        50, dg.initial_condition.shape[1]), (
            "Incorrect forecast shape generated for {model}"
    )
    assert data.interv_states.shape == (50, dg.initial_condition.shape[1]), (
        "Incorrect forecast shape generated for {model}"
    )

    data = dg.generate_data(50, 50, num_burn_in_states=25)
    assert data.train_states.shape == (50, dg.initial_condition.shape[1]), (
        "Incorrect train shape generated for {model} with burn in."
    )
    assert data.forecast_states.shape == (
        50, dg.initial_condition.shape[1]), (
            "Incorrect forecast shape generated for {model} with burn in."
    )
    assert data.interv_states.shape == (50, dg.initial_condition.shape[1]), (
        "Incorrect forecast shape generated for {model} with burn in."
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