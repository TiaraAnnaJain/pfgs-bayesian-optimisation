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


def test_beta_crossing_bounds_return_nan():
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
        ([0.5, 0.8], [2.0, np.inf], (0.8, 2.0)),  # inf means "no upper bound"
        ([0.5, 0.8], [np.inf, np.inf], (0.8, np.inf)),
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


def test_beta_without_upper_bound_is_inf():
    # The selected point has the highest standard deviation on the front.
    pareto = np.array([[1.0, 0.2], [0.8, 0.5]])

    beta_min, beta_max = calculate_beta(pareto, np.array([[0.8, 0.5]]))

    assert np.isclose(beta_min, (1.0 - 0.8) / (0.5 - 0.2))
    assert beta_max == np.inf


def test_beta_without_lower_bound_is_zero():
    # The selected point has the lowest standard deviation on the front.
    pareto = np.array([[1.0, 0.2], [0.5, 0.8]])

    beta_min, beta_max = calculate_beta(pareto, np.array([[1.0, 0.2]]))

    assert beta_min == 0.0
    assert np.isclose(beta_max, (0.5 - 1.0) / (0.2 - 0.8))


def test_beta_is_nan_when_another_point_has_same_std_and_higher_mean():
    pareto = np.array([[1.0, 0.5], [0.8, 0.5]])

    beta_min, beta_max = calculate_beta(pareto, np.array([[0.8, 0.5]]))

    assert np.isnan(beta_min) and np.isnan(beta_max)


def test_schedules_ignore_unbounded_and_invalid_intervals():
    schedules = beta_ranges(
        {"beta_min": [1.0, 2.0, np.nan], "beta_max": [3.0, np.inf, np.nan]}
    )

    # Only the first interval has a finite midpoint (2.0).
    np.testing.assert_allclose(schedules["mean"], 2.0)
    np.testing.assert_allclose(schedules["median"], 2.0)
    assert np.isinf(schedules["max"][1]) and np.isnan(schedules["max"][2])
    # [2, 3] is covered by both valid intervals, so max overlap picks 2.5.
    np.testing.assert_allclose(schedules["max_overlap"], 2.5)


def test_max_overlap_uses_lower_end_of_unbounded_range():
    schedules = beta_ranges({"beta_min": [1.0, 2.0], "beta_max": [np.inf, np.inf]})

    # Both intervals cover [2, inf), so the smallest beta in that range is used.
    np.testing.assert_allclose(schedules["max_overlap"], 2.0)
