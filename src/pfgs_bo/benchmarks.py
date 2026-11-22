"""
Registry of the benchmark problems: objective function, search space and optima.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pfgs_bo.data.investigation_space import (
    InvestigationSpace,
    InvestigationSpaceVariable,
)
from pfgs_bo.data.simulation import ackley, branin, himmelblau


@dataclass(frozen=True)
class Benchmark:
    """
    A benchmark problem.

    Parameters:
        name (str): problem name used in configs and on the command line.
        function (Callable): objective function to maximise.
        space (InvestigationSpace): search space and output label.
        maxima (list[list[float]]): locations of the global maxima, used to build
            the design-space test set around them.
    """

    name: str
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    space: InvestigationSpace
    maxima: list[list[float]]


def _space(bounds: dict[str, tuple[float, float]]) -> InvestigationSpace:
    """Builds a search space with output "y" from {label: (lower, upper)}."""
    return InvestigationSpace(
        variables=[
            InvestigationSpaceVariable(
                label=label, lower_bound=lower, upper_bound=upper
            )
            for label, (lower, upper) in bounds.items()
        ],
        outputs=["y"],
    )


def _ackley(dimensions: int) -> Benchmark:
    """Ackley in the given number of dimensions, with its maximum at the origin."""
    bounds = (-32.768, 32.768)
    return Benchmark(
        name="ackley",
        function=ackley,
        space=_space({f"x{i}": bounds for i in range(1, dimensions + 1)}),
        maxima=[[0.0] * dimensions],
    )


BENCHMARKS: dict[str, Benchmark] = {
    "himmelblau": Benchmark(
        name="himmelblau",
        function=himmelblau,
        space=_space({"x1": (-5.0, 5.0), "x2": (-5.0, 5.0)}),
        maxima=[
            [3.0, 2.0],
            [-2.805118, 3.131312],
            [-3.779310, -3.283186],
            [3.584428, -1.848126],
        ],
    ),
    "branin": Benchmark(
        name="branin",
        function=branin,
        space=_space({"x1": (-5.0, 10.0), "x2": (0.0, 15.0)}),
        maxima=[[-3.142, 12.275], [3.142, 2.275], [9.425, 2.475]],
    ),
    "ackley": _ackley(2),
}

VARIABLE_DIMENSIONS = {"ackley"}


def get_benchmark(name: str, dimensions: int = 2) -> Benchmark:
    """
    Returns the benchmark with the given name and number of input dimensions.

    Parameters:
        name (str): problem name.
        dimensions (int): number of input dimensions; only problems in
            ``VARIABLE_DIMENSIONS`` support values other than 2.

    Returns:
        benchmark (Benchmark): the benchmark.

    Raises:
        ValueError: if there is no benchmark with that name, or it doesn't support
            the number of dimensions.
    """
    if name not in BENCHMARKS:
        raise ValueError(
            f"Problem {name!r} is not supported. Choose from: {', '.join(BENCHMARKS)}."
        )
    if dimensions == 2:
        return BENCHMARKS[name]
    if name not in VARIABLE_DIMENSIONS:
        raise ValueError(f"Problem {name!r} is only defined in 2 dimensions.")
    if dimensions < 1:
        raise ValueError("Number of dimensions must be at least 1.")
    return _ackley(dimensions)
