from typing import Type

import pytest
import interfere_experiments as ie


@pytest.mark.parametrize("data_gen_type", ie.data_generators.ALL_MODELS)
class TestDataGenerators:

    def test_generate(
        self,
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


