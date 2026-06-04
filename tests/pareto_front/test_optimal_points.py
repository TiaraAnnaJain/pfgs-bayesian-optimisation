"""Tests for optimal point calculator."""

import numpy as np
from src.pareto_front.optimal_point import OptimalPoints


def test_closest_to_utopia_point():
    array = np.array([[0.2, 0.2], [0.8, 0.8], [0.5, 0.5], [0.9, 0.8]])
    array_unscaled = array.copy()
    utopia_point = np.array([1.0, 1.0])

    calculator = OptimalPoints(array, array_unscaled, utopia_point)

    closest_point, closest_idx, _ = calculator.closest_points(array)

    assert np.allclose(closest_point, [[0.9, 0.8]])
    assert closest_idx == 3
