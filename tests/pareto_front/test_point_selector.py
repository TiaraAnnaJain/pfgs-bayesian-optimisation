"""Tests for optimal point calculator."""

import numpy as np
from src.pareto_front.optimal_point import ParetoPointSelector


def test_closest_to_utopia_point():
    array = np.array([[0.2, 0.2], [0.8, 0.8], [0.5, 0.5], [0.9, 0.8]])
    array_unscaled = array.copy()
    utopia_point = np.array([1.0, 1.0])

    calculator = ParetoPointSelector(array, array_unscaled, utopia_point)

    closest_point, closest_idx, _ = calculator.closest_points(array, array_unscaled)

    assert np.allclose(closest_point, [[0.9, 0.8]])
    assert closest_idx == 3


def test_discard_areas():
    array = np.array([[0.2, 0.2], [0.8, 0.8], [0.5, 0.5], [0.9, 0.8]])
    array_unscaled = array.copy()
    utopia_point = np.array([0, 0])

    mean_discard_threshold = np.array([0.3])
    variance_discard_threshold = np.array([0.3])

    calculator = ParetoPointSelector(array, array_unscaled, utopia_point)
    _, array_scaled, _, _ = calculator.discard_areas(
        mean_discard_threshold=mean_discard_threshold,
        variance_discard_threshold=variance_discard_threshold,
    )

    closest_point, _, _ = calculator.closest_points(array_scaled, array_unscaled)

    assert np.allclose(closest_point, [[0.5, 0.5]])
