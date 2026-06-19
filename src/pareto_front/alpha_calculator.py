"""Calculator to find UCB coefficient alpha of the optimal point on pareto front."""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray


def calculate_alpha(
    pareto_front: NDArray[np.float64], closest_point: NDArray[np.float64]
) -> Tuple[float, float]:
    """
    Computes the range of α values that maximise the selected optimal point on the
    Pareto front for a UCB acquisition function.

     Parameters:
         pareto_front (NDArray[np.float64]): pareto front points.
         closest_point (NDArray[np.float64]): optimal point on the pareto front.

     Returns:
         alpha_min (float), alpha_max (float): minimum and maximum
             alpha values.
    """
    optimal_mean = float(closest_point[0, 0])
    optimal_std_dev = float(closest_point[0, 1])

    lower_bounds = []
    upper_bounds = []

    for mean, std_dev in pareto_front:
        if std_dev < optimal_std_dev:
            alpha_lower = (mean - optimal_mean) / (optimal_std_dev - std_dev)
            lower_bounds.append(alpha_lower)
        elif std_dev > optimal_std_dev:
            alpha_upper = (mean - optimal_mean) / (optimal_std_dev - std_dev)
            upper_bounds.append(alpha_upper)
        else:
            if optimal_mean < mean:
                return 0, 0
            continue

    alpha_min = float(max(lower_bounds)) if lower_bounds else 0
    alpha_max = float(min(upper_bounds)) if upper_bounds else 0

    if alpha_min > alpha_max:
        return np.nan, np.nan

    return alpha_min, alpha_max
