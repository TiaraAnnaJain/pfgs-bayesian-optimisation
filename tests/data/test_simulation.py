"""Tests for the benchmark objective functions."""

import numpy as np
import pytest

from pfgs_bo.data.simulation import SIMULATION_REGISTRY


@pytest.mark.parametrize("name", sorted(SIMULATION_REGISTRY))
def test_simulations_return_one_value_per_point(name):
    x = np.array([[0.0, 0.0], [1.0, -1.0], [2.5, 3.0]])

    y = SIMULATION_REGISTRY[name](x)

    assert y.shape == (3,)
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("name", sorted(SIMULATION_REGISTRY))
def test_simulations_reject_non_2d_input(name):
    with pytest.raises(ValueError, match="2-dimensional"):
        SIMULATION_REGISTRY[name](np.zeros((4, 3)))
