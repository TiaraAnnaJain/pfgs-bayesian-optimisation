"""Functions to compute metrics."""

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error


def compute_rmse(
    predictions_mean: NDArray[np.float64], y_true: NDArray[np.float64]
) -> float:
    """
    Computes the RMSE of the prediction mean values.

    Parameters:
        predictions_mean (NDArray[np.float64]): prediction mean values from GP.
        y_true (NDArray[np.float64]): true y values from test set.

    Returns:
        rmse (float): The RMSE of the prediction mean values.
    """
    rmse = np.sqrt(mean_squared_error(y_true, predictions_mean))
    return rmse


def compute_rrmse(
    predictions_mean: NDArray[np.float64], y_true: NDArray[np.float64]
) -> float:
    """
    Computes the relative RMSE of the prediction mean values.

    Parameters:
        predictions_mean (NDArray[np.float64]): prediction mean values from GP.
        y_true (NDArray[np.float64]): true y values from test set.

    Returns:
        rrmse (float): The RRMSE of the prediction mean values.
    """
    std_dev = np.std(y_true)
    rmse = compute_rmse(predictions_mean, y_true)
    rrmse = rmse / std_dev

    return rrmse
