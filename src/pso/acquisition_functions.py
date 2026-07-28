"""UCB and EI acquisition functions for Bayesian Optimisation."""

import numpy as np
from numpy.typing import NDArray
import torch
from torch.distributions import Normal

from data.data_models import Dataset


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
