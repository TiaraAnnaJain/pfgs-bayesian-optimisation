"""Generates datasets using LHS sampling."""

from typing import Tuple, Union, List

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from src.data.data_models import Dataset
from src.data.simulation import himmelblau


def lhs_sampler(
    n_samples: int,
    n_vars: int,
    lower_bound: List[float],
    upper_bound: List[float],
    seed: int,
) -> NDArray[np.float64]:
    """
    LHS sampler function to generate datasets.

    Parameters:
        n_samples (int): Number of samples to generate.
        n_vars (int): Number of dimensions.
        lower_bound (List[float]): Lower bound.
        upper_bound (List[float]): Upper bound.
        seed (int): Random seed for reproducibility.

    Returns:
        x_scaled (NDArray[np.float64]): sampled and scaled dataset.
    """
    sampler = qmc.LatinHypercube(d=n_vars, seed=seed)
    x_unit = sampler.random(n=n_samples)
    x_scaled = qmc.scale(x_unit, lower_bound, upper_bound)
    return x_scaled


def generate_data(
    seed_train: int = 42,
    seed_test: int = 10,
    delta: int = 1.5,
    maxima: Union[NDArray[np.float64], None] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Uses LHS sampling to generate training and test datasets.

    Parameters:
        seed_train (int): random seed for reproducibility of training
            data
        seed_test (int): random seed for reproducibility of test data
        delta (float): delta parameter for LHS bounds around maxima.
        maxima (Union[NDArray[np.float64], None]): maxima values of the Himmelblau function.

    Returns:
        Dataset(x_train, y_train): training dataset
        Dataset(x_test, y_test): test dataset
        Dataset(x_test_design_space_set,yx_test_design_space_set): test dataset in the
        design space only.
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

    x_train = lhs_sampler(
        n_samples=5,
        n_vars=n_vars,
        lower_bound=l_bounds,
        upper_bound=u_bounds,
        seed=seed_train,
    )
    x_test = lhs_sampler(
        n_samples=100,
        n_vars=n_vars,
        lower_bound=l_bounds,
        upper_bound=u_bounds,
        seed=seed_test,
    )

    y_train = np.array([himmelblau(x_train)]).reshape(-1, 1)
    y_test = np.array([himmelblau(x_test)]).reshape(-1, 1)

    x_test_design_space = []
    y_test_design_space = []

    for m in maxima:
        lower_bound = m - delta
        upper_bound = m + delta

        x_test_local = lhs_sampler(
            n_samples=25,
            n_vars=n_vars,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            seed=seed_test,
        )
        y_test_local = np.array([himmelblau(x_test_local)]).reshape(-1, 1)
        x_test_design_space.append(x_test_local)
        y_test_design_space.append(y_test_local)

    x_test_design_space_set = np.vstack(x_test_design_space)
    y_test_design_space_set = np.vstack(y_test_design_space)

    return (
        Dataset(x_train, y_train),
        Dataset(x_test, y_test),
        Dataset(x_test_design_space_set, y_test_design_space_set),
    )
