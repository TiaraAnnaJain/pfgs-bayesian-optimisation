"""Tests for beta calculator"""

import numpy as np
import pytest

from pfgs_bo.pareto_front.beta import (
    beta_ranges,
    calculate_beta,
    feasible_beta_range,
    select_schedules,
)


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
        ([0.5, np.nan], [2.0, np.nan], (0.5, 2.0)),
        ([0.5, 0.8], [2.0, np.inf], (0.8, 2.0)),
        ([0.5, 0.8], [np.inf, np.inf], (0.8, np.inf)),
        ([1.0, 2.0], [1.5, 3.0], None),
        ([], [], None),
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
    np.testing.assert_allclose(schedules["max_overlap"], 1.1)


def test_beta_without_upper_bound_is_inf():
    pareto = np.array([[1.0, 0.2], [0.8, 0.5]])

    beta_min, beta_max = calculate_beta(pareto, np.array([[0.8, 0.5]]))

    assert np.isclose(beta_min, (1.0 - 0.8) / (0.5 - 0.2))
    assert beta_max == np.inf


def test_beta_without_lower_bound_is_zero():
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

    np.testing.assert_allclose(schedules["mean"], 2.0)
    np.testing.assert_allclose(schedules["median"], 2.0)
    assert np.isinf(schedules["max"][1]) and np.isnan(schedules["max"][2])
    np.testing.assert_allclose(schedules["max_overlap"], 2.5)


def test_max_overlap_uses_lower_end_of_unbounded_range():
    schedules = beta_ranges({"beta_min": [1.0, 2.0], "beta_max": [np.inf, np.inf]})

    np.testing.assert_allclose(schedules["max_overlap"], 2.0)


def test_select_schedules_picks_one_and_includes_max_overlap_ties():
    schedules = {"max": 1, "max_overlap": 2, "max_overlap_2": 3, "min": 4}

    assert select_schedules(schedules, None) == schedules
    assert select_schedules(schedules, "max") == {"max": 1}
    assert select_schedules(schedules, "max_overlap") == {
        "max_overlap": 2,
        "max_overlap_2": 3,
    }
    with pytest.raises(ValueError, match="Unknown beta schedule"):
        select_schedules(schedules, "horizontal")
