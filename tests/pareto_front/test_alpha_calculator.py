"""Tests for alpha calculator"""

import numpy as np
from src.pareto_front.alpha_calculator import AlphaCalculator


def test_alpha_valid_range():
    pareto = np.array([[1.0, 0.2], [0.5, 0.8]])

    optimal = np.array([[0.8, 0.5]])

    calc = AlphaCalculator(pareto, optimal)
    alpha_min, alpha_max = calc.calculate_alpha()

    expected_lower = (1.0 - 0.8) / (0.5 - 0.2)
    expected_upper = (0.5 - 0.8) / (0.5 - 0.8)

    assert np.isclose(alpha_min, expected_lower)
    assert np.isclose(alpha_max, expected_upper)
    assert alpha_min <= alpha_max


def test_alpha_invalid_range_returns_zero():
    pareto = np.array([[2.0, 0.2], [0.0, 0.8]])

    optimal = np.array([[1.0, 0.5]])

    calc = AlphaCalculator(pareto, optimal)
    alpha_min, alpha_max = calc.calculate_alpha()

    assert alpha_min == 0
    assert alpha_max == 0
