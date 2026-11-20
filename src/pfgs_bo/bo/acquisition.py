"""Acquisition functions (UCB, EI) and the prediction-space filter used by constrained BO."""

from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray
import torch
from torch.distributions import Normal

from pfgs_bo.data.data_models import Dataset


def compute_ucb(predictions: Dataset, beta: float) -> NDArray[np.float64]:
    """
    Computes the UCB function for a given mean and variance.

    Parameters:
        predictions (Dataset): mean and standard deviation predictions from GP output.
        beta (float): beta parameter

    Returns:
        ucb (NDArray[np.float64]): UCB function values
    """

    ucb = predictions.y_data + (predictions.y_errors * beta)
    return ucb


def compute_ei(predictions: Dataset, train_data: Dataset) -> NDArray[np.float64]:
    """
    Computes the expected improvement acquisition function for a given mean and variance.

    Parameters:
        predictions (Dataset): mean and standard deviation predictions from GP output.
        train_data (Dataset): training data for train outputs.

    Returns:
        ei (NDArray[np.float64]): EI function values.
    """
    std_tensor = torch.tensor(predictions.y_errors, dtype=torch.float64)
    mean_tensor = torch.tensor(predictions.y_data, dtype=torch.float64)

    normal = Normal(torch.zeros_like(mean_tensor), torch.ones_like(mean_tensor))

    train_y = torch.tensor(train_data.y_data, dtype=torch.float64)

    best_f = train_y.max()
    z = (mean_tensor - best_f) / std_tensor

    ei = ((mean_tensor - best_f) * normal.cdf(z)) + (
        std_tensor * normal.log_prob(z).exp()
    )
    ei = torch.where(std_tensor > 0, ei, torch.zeros_like(ei))
    ei = ei.numpy()

    return ei


def filter_predictions(
    preds: Dataset,
    mean_threshold: NDArray[np.float64],
    std_threshold: NDArray[np.float64],
    normalisation_stats: Dict[str, Any],
) -> NDArray[np.float64]:
    """
    Creates a filter based on thresholds to apply to acquisition function.

    Parameters:
        preds (Dataset): GP predictions.
        mean_threshold (NDArray[np.float64]): mean threshold.
        std_threshold (NDArray[np.float64]): standard deviation threshold.
        normalisation_stats (Dict[str, Any]): normalisation stats.

    Returns:
        mask (NDArray[np.float64]): boolean mask of predictions.
    """
    preds_raw = preds.copy()
    preds_raw.set_stats(normalisation_stats)
    preds_raw.de_normalise()

    mean = preds_raw.y_data
    std = preds_raw.y_errors

    mask = (mean >= mean_threshold) & (std >= std_threshold)
    mask = np.asarray(mask).reshape(-1)

    return mask
