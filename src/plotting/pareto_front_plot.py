"""Generates a plot of the pareto front with discard thresholds and optimal point"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

from src.data.data_models import Dataset


def plot_pareto_front(
    unscaled_pareto_front: NDArray[np.float64],
    train_data: Dataset,
    discard_thresholds: NDArray[np.float64],
    optimal_point: NDArray[np.float64],
    scaler: MinMaxScaler,
    iteration: int,
) -> plt.Figure:
    """
    Plots the pareto front with discard thresholds and optimal point

    Parameters:
        unscaled_pareto_front (NDArray[np.float64]): The unscaled, standardised pareto
            front.
        train_data (Dataset): The training dataset.
        discard_thresholds (NDArray[np.float64]): unscaled, standardised threshold
            below which mean and variance are discarded.
        optimal_point (NDArray[np.float64]): unscaled, standardised optimal point
            calculated.
        scaler (MinMaxScaler): scaler used to scale pareto front points.
        iteration (int): current iteration, passed into plot title.
    """
    pareto_front = train_data.copy()
    pareto_front.y_data = unscaled_pareto_front[:, 0].reshape(-1, 1).copy()
    pareto_front.y_errors = unscaled_pareto_front[:, 1].reshape(-1, 1).copy()
    pareto_front.dataset_type = "pareto_front"
    pareto_front.normalised = True
    pareto_front.de_normalise()

    optimal_point_destandardised = optimal_point.copy()
    optimal_point_destandardised[:, 0] = (
        optimal_point[:, 0] * train_data.y_standardiser_std["y"]
        + train_data.y_standardiser_mean["y"]
    )
    optimal_point_destandardised[:, 1] = (
        optimal_point[:, 1] * train_data.y_standardiser_std["y"] ** 2
    )

    discard_thresholds = scaler.inverse_transform(discard_thresholds.reshape(1, -1))

    discard_thresholds_destandardised = discard_thresholds.copy()
    discard_thresholds_destandardised[:, 0] = (
        discard_thresholds[:, 0] * train_data.y_standardiser_std["y"]
        + train_data.y_standardiser_mean["y"]
    )
    discard_thresholds_destandardised[:, 1] = (
        discard_thresholds[:, 1] * train_data.y_standardiser_std["y"] ** 2
    )

    pareto_plot = plt.figure()

    plt.scatter(
        pareto_front.y_data,
        pareto_front.y_errors,
        s=30,
        label="Pareto front",
    )
    plt.axvline(
        discard_thresholds_destandardised[0, 0],
        linestyle="--",
        color="red",
        label="Mean discard threshold",
    )
    plt.axhline(
        discard_thresholds_destandardised[0, 1],
        linestyle="--",
        color="red",
        label="Variance discard threshold",
    )
    plt.scatter(
        optimal_point_destandardised[:, 0],
        optimal_point_destandardised[:, 1],
        color="red",
        label="Optimal point",
        zorder=5,
    )
    plt.title(f"Pareto front at iteration {iteration}")
    plt.xlabel("Mean")
    plt.ylabel("Variance")
    plt.legend()

    return pareto_plot
