"""UCB beta derivation from the Pareto front and beta schedules for PFGS-guided BO."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def calculate_beta(
    pareto_front: NDArray[np.float64], closest_point: NDArray[np.float64]
) -> tuple[float, float]:
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


def most_overlapping_beta(
    beta_min: NDArray[np.float64], beta_max: NDArray[np.float64]
) -> tuple[NDArray[np.float64], int]:
    """
    Finds the beta values that lie inside the largest number of PFGS beta intervals.

    Sweeps over the interval end points, counting how many intervals are open
    between each pair of consecutive end points. Intervals with a non-finite bound
    are ignored.

    Parameters:
        beta_min (NDArray[np.float64]): lower bound of the beta interval per iteration.
        beta_max (NDArray[np.float64]): upper bound of the beta interval per iteration.

    Returns:
        best_betas (NDArray[np.float64]): midpoint of each beta range covered by the
            most intervals (adjacent ranges are merged); empty if there are none.
        best (int): number of intervals covering those ranges.
    """
    events = []

    for lower, upper in zip(beta_min, beta_max, strict=True):
        if np.isfinite(lower) and np.isfinite(upper):
            events.append((lower, +1))
            events.append((upper, -1))

    if not events:
        return np.array([]), 0

    events.sort(key=lambda x: (x[0], -x[1]))

    current = 0
    best = 0
    best_ranges = []

    for i in range(len(events) - 1):
        beta, delta = events[i]
        current += delta
        next_beta = events[i + 1][0]

        if next_beta == beta:
            continue

        if current > best:
            best = current
            best_ranges = [(beta, next_beta)]

        elif current == best and best > 0:
            if best_ranges and np.isclose(best_ranges[-1][1], beta):
                best_ranges[-1] = (best_ranges[-1][0], next_beta)
            else:
                best_ranges.append((beta, next_beta))

    if best <= 0 or not best_ranges:
        return np.array([]), 0

    best_betas = [(a + b) / 2 for a, b in best_ranges]

    return np.array(best_betas), best


def beta_ranges(
    beta_range: dict[str, list[float]],
) -> dict[str, NDArray[np.float64]]:
    """
    Takes a dictionary of beta ranges and returns a dictionary of beta schedules with
    the min, max, mid-point, mean and meadian.
    Parameters:
        beta_range (dict): Dictionary of beta ranges from json file.
    Returns:
         beta_schedules (dict): Dictionary of beta schedules.
    """
    beta_min = np.array(beta_range["beta_min"])
    beta_max = np.array(beta_range["beta_max"])

    mid_point = (beta_max + beta_min) / 2
    mean_value = np.nanmean(mid_point)
    median_value = np.nanmedian(mid_point)

    horizontal_betas, best = most_overlapping_beta(beta_min, beta_max)

    beta_schedule = {
        "min": beta_min,
        "max": beta_max,
        "mid_point": mid_point,
        "mean": np.full(len(beta_min), mean_value),
        "median": np.full(len(beta_min), median_value),
    }

    beta_schedule.update(
        {
            f"horizontal ({beta:.3f})": np.full(len(beta_min), beta)
            for beta in horizontal_betas
        }
    )

    return beta_schedule


def feasible_beta_range(
    beta_lower: Sequence[float], beta_upper: Sequence[float]
) -> tuple[float, float] | None:
    """
    Finds the beta range that lies inside the beta interval of every PFGS iteration.
    Iterations without a valid interval (NaN) are skipped, and an upper bound of 0
    is treated as "no upper bound".

    Parameters:
        beta_lower (Sequence[float]): lower bound of the beta interval per iteration.
        beta_upper (Sequence[float]): upper bound of the beta interval per iteration.

    Returns:
        feasible_range (tuple[float, float] | None): (min, max) of the common range,
            or None if the intervals don't overlap or there are none.
    """
    lower = [beta for beta in beta_lower if np.isfinite(beta)]
    upper = [beta for beta in beta_upper if np.isfinite(beta) and beta != 0]

    if not lower or not upper:
        return None

    beta_min, beta_max = max(lower), min(upper)
    if beta_max < beta_min:
        return None

    return beta_min, beta_max
