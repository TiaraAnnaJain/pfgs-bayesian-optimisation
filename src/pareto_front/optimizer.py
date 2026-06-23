"""NSGA-2 optimizer to generate pareto front"""

from typing import Any, Tuple

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import numpy as np
from numpy.typing import NDArray


class NSGAOptimiser:
    def __init__(self, problem: Any, pop_size: int = 500, n_gen: int = 10) -> None:
        self.problem = problem
        self.algorithm = NSGA2(pop_size)
        self.termination = get_termination("n_gen", n_gen)

    def run(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Run the multi-objective optimiser and compute the Pareto front.

        Returns:
            pareto_front (NDArray[np.float64]): pareto front points.
            results.X (NDArray[np.float64]): input values.

        """
        results = minimize(self.problem, self.algorithm, self.termination)

        pareto_front = -results.F
        return (
            pareto_front,
            results.X,
        )
