"""Surrogate accuracy metrics (RMSE, RRMSE) and evaluation on test sets."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error

from pfgs_bo.data.data_models import Dataset


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
    Computes the relative RMSE of the prediction mean values: the RMSE divided by the
    (population) standard deviation of the true values. An RRMSE of 1 is as good as
    always predicting the mean of the test set.

    Parameters:
        predictions_mean (NDArray[np.float64]): prediction mean values from GP.
        y_true (NDArray[np.float64]): true y values from test set.

    Returns:
        rrmse (float): The RRMSE of the prediction mean values.
    """
    std_dev = np.std(y_true)
    rmse = compute_rmse(predictions_mean, y_true)
    rrmse = rmse / std_dev

    return float(rrmse)


def calculate_model_metrics(
    model: Any, datasets: dict[str, Dataset], normalisation_stats: dict[str, dict]
) -> dict[str, float]:
    """
    Computes the RRMSE of the model on each test set.

    Parameters:
        model (Any): fitted GP model.
        datasets (dict[str, Dataset]): de-normalised test sets, keyed by name.
        normalisation_stats (dict[str, Dict]): statistics used to normalise the
            training data.

    Returns:
        metrics (dict[str, float]): "<name> RRMSE" for each test set.
    """
    metrics = {}

    for name, dataset in datasets.items():
        recipe_dataset = dataset.copy()
        recipe_dataset.dataset_type = "recipe"
        recipe_dataset.y_data = None
        recipe_dataset.y_labels = None

        recipe_dataset.set_stats(normalisation_stats)
        recipe_dataset.normalise()

        predictions = model.predict(recipe_dataset)

        predictions.set_stats(normalisation_stats)
        predictions.de_normalise()

        metrics[f"{name} RRMSE"] = compute_rrmse(
            predictions.y_data.reshape(-1, 1), dataset.y_data
        )

    return metrics
