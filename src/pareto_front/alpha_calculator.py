"""Calculator to find UCB coefficient alpha of the optimal point on pareto front."""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray


class AlphaCalculator:
    """
    Computes the range of α values that maximise the selected optimal point on the
    Pareto front for a UCB acquisition function.

    Parameters:
        pareto_front (NDArray[np.float64]): pareto front points.
        closest_point (NDArray[np.float64]): optimal point on the pareto front.
    """

    def __init__(
        self, pareto_front: NDArray[np.float64], closest_point: NDArray[np.float64]
    ) -> None:
        self.pareto_front = pareto_front
        self.optimal_mean = float(closest_point[0, 0])
        self.optimal_var = float(closest_point[0, 1])

    def calculate_alpha(self) -> Tuple[float, float]:
        """
        Calculates alpha.

        Returns:
            alpha_min (float), alpha_max (float): minimum and maximum
                alpha values.
        """
        lower_bounds = []
        upper_bounds = []

        for mean, var in self.pareto_front:
            if var < self.optimal_var:
                alpha_lower = (mean - self.optimal_mean) / (self.optimal_var - var)
                lower_bounds.append(alpha_lower)
            elif var > self.optimal_var:
                alpha_upper = (mean - self.optimal_mean) / (self.optimal_var - var)
                upper_bounds.append(alpha_upper)
            else:
                if self.optimal_mean < mean:
                    return 0, 0
                continue

        alpha_min = float(max(lower_bounds)) if lower_bounds else 0
        alpha_max = float(min(upper_bounds)) if upper_bounds else 0

        if alpha_min > alpha_max:
            return 0, 0
        return alpha_min, alpha_max
