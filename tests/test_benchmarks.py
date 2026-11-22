"""Tests for the benchmark registry."""

import numpy as np
import pytest
from scipy.optimize import minimize

from pfgs_bo.benchmarks import BENCHMARKS, get_benchmark


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_maxima_lie_inside_the_search_space(name):
    benchmark = BENCHMARKS[name]
    lower = np.array(benchmark.space.lower_bound_list)
    upper = np.array(benchmark.space.upper_bound_list)

    for maximum in benchmark.maxima:
        assert np.all(lower <= maximum) and np.all(np.array(maximum) <= upper)


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_each_listed_maximum_is_a_local_maximum(name):
    benchmark = BENCHMARKS[name]

    for maximum in benchmark.maxima:
        result = minimize(
            lambda x: -benchmark.function(x.reshape(1, -1))[0],
            maximum,
            method="Nelder-Mead",
        )
        # The design-space test regions are +/- delta (1.5) around these points.
        assert np.linalg.norm(result.x - maximum) < 0.1


def test_unknown_problem_is_rejected():
    with pytest.raises(ValueError, match="not supported"):
        get_benchmark("rosenbrock")
