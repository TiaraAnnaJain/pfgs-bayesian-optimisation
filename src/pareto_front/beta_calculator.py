"""Calculator to find UCB coefficient beta of the optimal point on pareto front."""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray


def calculate_beta(
    pareto_front: NDArray[np.float64], closest_point: NDArray[np.float64]
) -> Tuple[float, float]:
    """
    Computes the range of α values that maximise the selected optimal point on the
    Pareto front for a UCB acquisition function.

     Parameters:
         pareto_front (NDArray[np.float64]): pareto front points.
         closest_point (NDArray[np.float64]): optimal point on the pareto front.

     Returns:
         beta_min (float), beta_max (float): minimum and maximum
             beta values.
    """
    optimal_mean = float(closest_point[0, 0])
    optimal_std_dev = float(closest_point[0, 1])

    lower_bounds = []
    upper_bounds = []

    for mean, std_dev in pareto_front:
        if std_dev < optimal_std_dev:
            beta_lower = (mean - optimal_mean) / (optimal_std_dev - std_dev)
            lower_bounds.append(beta_lower)
        elif std_dev > optimal_std_dev:
            beta_upper = (mean - optimal_mean) / (optimal_std_dev - std_dev)
            upper_bounds.append(beta_upper)
        else:
            if optimal_mean < mean:
                return 0, 0
            continue

    beta_min = float(max(lower_bounds)) if lower_bounds else 0
    beta_max = float(min(upper_bounds)) if upper_bounds else 0

    if beta_min > beta_max:
        return np.nan, np.nan

    return beta_min, beta_max
