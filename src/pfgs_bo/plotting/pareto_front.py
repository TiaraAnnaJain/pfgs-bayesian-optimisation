"""Generates a plot of the pareto front with discard thresholds and optimal point"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from pfgs_bo.data.data_models import Dataset


def plot_pareto_front(
    pareto_front: Dataset,
    discard_thresholds: NDArray[np.float64],
    optimal_point: NDArray[np.float64],
    iteration: int,
) -> plt.Figure:
    """
    Plots the pareto front with discard thresholds and optimal point

    Parameters:
        pareto_front (Dataset): unscaled, de-normalised pareto front.
        discard_thresholds (NDArray[np.float64]): unscaled threshold below which mean
            and standard deviation are discarded.
        optimal_point (NDArray[np.float64]): unscaled optimal point calculated.
        iteration (int): current iteration, passed into plot title.

    Returns:
        pareto_plot (matplotlib.figure.Figure): pareto front plot.
    """
    pareto_plot = plt.figure()

    plt.scatter(
        pareto_front.y_data,
        pareto_front.y_errors,
        s=30,
        label="Pareto front",
    )
    plt.axvline(
        float(discard_thresholds[0]),
        linestyle="--",
        color="red",
        label="Mean discard threshold",
    )
    plt.axhline(
        float(discard_thresholds[1]),
        linestyle="--",
        color="red",
        label="Standard deviation discard threshold",
    )
    plt.scatter(
        optimal_point[:, 0],
        optimal_point[:, 1],
        color="red",
        label="Optimal point",
        zorder=5,
    )
    plt.title(f"Pareto front at iteration {iteration}")
    plt.xlabel("Mean")
    plt.ylabel("Standard deviation")
    plt.legend()

    return pareto_plot
