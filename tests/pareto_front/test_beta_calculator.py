"""Tests for beta calculator"""

import numpy as np

from pareto_front.beta_calculator import calculate_beta


def test_beta_valid_range():
    pareto = np.array([[1.0, 0.2], [0.5, 0.8]])

    optimal = np.array([[0.8, 0.5]])

    beta_min, beta_max = calculate_beta(pareto, optimal)

    expected_lower = (1.0 - 0.8) / (0.5 - 0.2)
    expected_upper = (0.5 - 0.8) / (0.5 - 0.8)

    assert np.isclose(beta_min, expected_lower)
    assert np.isclose(beta_max, expected_upper)
    assert beta_min <= beta_max


def test_beta_invalid_range_returns_zero():
    pareto = np.array([[2.0, 0.2], [0.0, 0.8]])

    optimal = np.array([[1.0, 0.5]])

    beta_min, beta_max = calculate_beta(pareto, optimal)

    assert np.isnan(beta_min)
    assert np.isnan(beta_max)
