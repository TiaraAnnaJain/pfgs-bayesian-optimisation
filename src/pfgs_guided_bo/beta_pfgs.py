"""Functions to return beta schedules for BO."""

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


def most_overlapping_beta(beta_min, beta_max):
    events = []

    for l, u in zip(beta_min, beta_max):
        if np.isfinite(l) and np.isfinite(u):
            events.append((l, +1))
            events.append((u, -1))

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
    beta_range: Dict[str, List[float]],
) -> Dict[str, NDArray[np.float64]]:
    """
    Takes a dictionary of beta ranges and returns a dictionary of beta schedules with
    the min, max, mid-point, mean and meadian.
    Parameters:
        beta_range (dict): Dictionary of beta ranges from json file.
    Returns:
         beta_schedules (dict): Dictionary of beta schedules.
    """
    beta_min = np.array(beta_range["beta_min"])
    beta_max = beta_range["beta_max"]

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
