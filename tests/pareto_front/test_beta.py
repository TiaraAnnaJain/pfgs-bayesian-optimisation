"""Tests for beta calculator"""

import numpy as np
import pytest

from pfgs_bo.pareto_front.beta import beta_ranges, calculate_beta, feasible_beta_range


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


@pytest.mark.parametrize(
    ("beta_lower", "beta_upper", "expected"),
    [
        ([0.5, 0.8], [2.0, 1.5], (0.8, 1.5)),
        ([0.5, np.nan], [2.0, np.nan], (0.5, 2.0)),  # NaN iterations are skipped
        ([0.5, 0.8], [2.0, 0.0], (0.8, 2.0)),  # 0 means "no upper bound"
        ([1.0, 2.0], [1.5, 3.0], None),  # intervals don't overlap
        ([], [], None),  # PFGS stopped before the first beta
    ],
)
def test_feasible_beta_range(beta_lower, beta_upper, expected):
    assert feasible_beta_range(beta_lower, beta_upper) == expected


def test_beta_ranges_names_the_six_schedules():
    schedules = beta_ranges({"beta_min": [0.5, 0.8, 1.0], "beta_max": [2.0, 1.5, 1.2]})

    assert set(schedules) == {
        "min",
        "max",
        "mid_point",
        "mean",
        "median",
        "max_overlap",
    }
    # All three intervals contain [1.0, 1.2], so maximum overlap picks its midpoint.
    np.testing.assert_allclose(schedules["max_overlap"], 1.1)
