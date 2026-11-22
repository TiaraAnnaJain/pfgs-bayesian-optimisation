"""
Registry of the benchmark problems: objective function, search space and optima.

A config only names its problem (``problem = "himmelblau"``); everything that is a
fact about the problem itself lives here, so switching problem from the command line
(``--problem branin``) needs no other change. Settings that are experiment choices,
such as the discard thresholds, stay in the config.
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
    "ackley": Benchmark(
        name="ackley",
        function=ackley,
        space=_space({"x1": (-32.768, 32.768), "x2": (-32.768, 32.768)}),
        maxima=[[0.0, 0.0]],
    ),
}


def get_benchmark(name: str) -> Benchmark:
    """
    Returns the benchmark with the given name.

    Parameters:
        name (str): problem name.

    Returns:
        benchmark (Benchmark): the benchmark.

    Raises:
        ValueError: if there is no benchmark with that name.
    """
    if name not in BENCHMARKS:
        raise ValueError(
            f"Problem {name!r} is not supported. Choose from: {', '.join(BENCHMARKS)}."
        )
    return BENCHMARKS[name]
