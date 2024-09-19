from typing import Callable

import interfere_experiments
import pytest

QUICK_MODELS = [
    interfere_experiments.quick_models.gut_check_belozyorov, 
    interfere_experiments.quick_models.gut_check_coupled_logistic
]

@pytest.mark.parametrize("model_func", QUICK_MODELS)
def test_initialization(model_func: Callable):
    """Tests that quick models initialize."""

    model = model_func()
    assert model is not None
