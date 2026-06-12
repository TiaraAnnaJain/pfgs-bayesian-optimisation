"""NSGA-2 optimizer to generate pareto front"""

from typing import Any, Tuple

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy.typing import NDArray


class NSGAOptimiser:
    def __init__(self, problem: Any, pop_size: int = 500, n_gen: int = 10) -> None:
        self.problem = problem
        self.algorithm = NSGA2(pop_size)
        self.termination = get_termination("n_gen", n_gen)

    def run(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], MinMaxScaler]:
        """
        Run the multi-objective optimiser and compute the Pareto front.

        Returns:
            scaled_pareto_front_points (NDArray[np.float64]): scaled pareto front points.
            unscaled_results (NDArray[np.float64]): unscaled pareto front points.
            results.X (NDArray[np.float64]): input values.
            scaler (MinMaxScaler): min-max scaler used to scale pareto front points.

        """
        results = minimize(self.problem, self.algorithm, self.termination)

        unscaled_pareto_front_points = -results.F
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_pareto_front_points = scaler.fit_transform(unscaled_pareto_front_points)
        return scaled_pareto_front_points, unscaled_pareto_front_points, results.X, scaler
