"""Tests for the benchmark objective functions."""

import numpy as np
import pytest

from pfgs_bo.data.simulation import SIMULATION_REGISTRY, ackley


@pytest.mark.parametrize("name", sorted(SIMULATION_REGISTRY))
def test_simulations_return_one_value_per_point(name):
    x = np.array([[0.0, 0.0], [1.0, -1.0], [2.5, 3.0]])

    y = SIMULATION_REGISTRY[name](x)

    assert y.shape == (3,)
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("name", ["himmelblau", "branin"])
def test_2d_simulations_reject_other_dimensions(name):
    with pytest.raises(ValueError, match="2-dimensional"):
        SIMULATION_REGISTRY[name](np.zeros((4, 3)))


def test_ackley_matches_the_original_2d_formula():
    x = np.array([[0.3, -1.7], [2.0, 4.5]])
    x1, x2 = x[:, 0], x[:, 1]
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))

    np.testing.assert_allclose(ackley(x), 25 - (term1 + term2 + np.e + 20))


@pytest.mark.parametrize("dimensions", [1, 2, 5, 10])
def test_ackley_maximum_is_25_at_the_origin(dimensions):
    origin = np.zeros((1, dimensions))
    rng = np.random.default_rng(0)
    others = rng.uniform(-32.768, 32.768, size=(500, dimensions))

    assert ackley(origin)[0] == pytest.approx(25)
    assert np.all(ackley(others) < 25)
