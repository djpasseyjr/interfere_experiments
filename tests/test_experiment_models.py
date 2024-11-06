from typing import Type

import pytest
import interfere_experiments as ie


@pytest.mark.parametrize("model_type", ie.experiment_models.ALL_MODELS)
class TestExperimentModels:

    def test_generate(self, model_type: Type[ie.experiment_models.ExperimentModel]):
        m = model_type()
        data = m.generate_data(50, 50)
        assert data.interv_states.shape == (50, m.initial_condition.shape[1]), (
            "Incorrect forecast shape generated for {model}"
        )


