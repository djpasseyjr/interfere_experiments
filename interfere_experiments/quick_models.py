import interfere
import numpy as np


def gut_check_coupled_logistic(
) -> interfere.dynamics.StochasticCoupledMapLattice:
    """Builds the coupled logistic map in the forecaster gut check notebook.

    Returns:
        An instance of interfere.dynamics.StochasticCoupledMapLattice.
    """
    
    return interfere.dynamics.coupled_logistic_map(
        **{
            # Two cycles and isolated node
            "adjacency_matrix": np.array([
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]),
            "eps": 0.9,
            "logistic_param": 3.72,
            "sigma": 0.0,
            "measurement_noise_std": 0.01 * np.ones(10),
    })


def gut_check_belozy() -> interfere.dynamics.Belozyorov3DQuad:
    """Builds the Belozyorov model used in the forecaster gut check notebook.

    Returns:
        An instance of interfere.dynamics.Belozyorov3DQuad.
    """
    return interfere.dynamics.Belozyorov3DQuad(
        mu=1.81, sigma=0.05, measurement_noise_std = 0.01 * np.ones(3),
    )