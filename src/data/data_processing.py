"""Generates datasets using LHS sampling."""

from typing import Tuple, Union
import numpy as np
from numpy.typing import NDArray
import torch
from scipy.stats import qmc
from src.data.data_models import Dataset


def himmelblau(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculates the modified Himmelblau function."""
    x_1, x_2 = x[:, 0], x[:, 1]
    y = ((x_1**2 + x_2 - 11) ** 2) + ((x_1 + x_2**2 - 7) ** 2) + x_1 - (2 * x_2)
    return np.exp(8 - 0.01 * y)


def generate_data(
    n_train: int = 5,
    n_test: int = 100,
    seed_train: int = 42,
    seed_test: int = 10,
    n_per_region: int = 25,
    delta: int = 1.5,
    maxima: Union[NDArray[np.float64], None] = None,
) -> Tuple[Dataset, Dataset, Dataset, int, int]:
    """
    Uses LHS sampling to generate training and test datasets.

    Parameters:
        n_train (int): number of training samples
        n_test (int): number of test samples
        seed_train (int): random seed for reproducibility of training
            data
        seed_test (int): random seed for reproducibility of test data
        n_per_region (int): number of test samples per region in design space test set.
        delta (float): delta parameter for LHS bounds around maxima.
        maxima (Union[NDArray[np.float64], None]): maxima values of the Himmelblau function.

    Returns:
        Dataset(x_train, y_train): training dataset
        Dataset(x_test, y_test): test dataset
        Dataset(x_test_design_space_set,yx_test_design_space_set): test dataset in the
        design space only.
        seed_train (int): random seed for reproducibility of training data
        seed_test (int): random seed for reproducibility of test data
    """

    if maxima is None:
        maxima = np.array(
            [
                [3.0, 2.0],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ]
        )

    # whole space
    n_vars = 2
    l_bounds = [-5, -5]
    u_bounds = [5, 5]

    train_sampler = qmc.LatinHypercube(d=n_vars, seed=seed_train)
    test_sampler = qmc.LatinHypercube(d=n_vars, seed=seed_test)

    x_train_unscaled = train_sampler.random(n=n_train)
    x_test_unscaled = test_sampler.random(n=n_test)

    x_train = qmc.scale(x_train_unscaled, l_bounds, u_bounds)
    x_test = qmc.scale(x_test_unscaled, l_bounds, u_bounds)

    y_train = np.array([himmelblau(x_train)]).reshape(-1, 1)
    y_test = np.array([himmelblau(x_test)]).reshape(-1, 1)

    x_test_design_space = []
    y_test_design_space = []

    for m in maxima:
        test_sampler = qmc.LatinHypercube(d=n_vars, seed=seed_test)
        x_test_unscaled = test_sampler.random(n=n_per_region)

        lower_bound = m - delta
        upper_bound = m + delta

        x_test_local = qmc.scale(x_test_unscaled, lower_bound, upper_bound)
        y_test_local = np.array([himmelblau(x_test_local)]).reshape(-1, 1)
        x_test_design_space.append(x_test_local)
        y_test_design_space.append(y_test_local)

    x_test_design_space_set = np.vstack(x_test_design_space)
    y_test_design_space_set = np.vstack(y_test_design_space)

    return (
        Dataset(x_train, y_train),
        Dataset(x_test, y_test),
        Dataset(x_test_design_space_set, y_test_design_space_set),
        seed_train,
        seed_test,
    )
