"""Generates datasets using LHS sampling."""

from typing import Union, List

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from src.data.data_models import Dataset
from src.data.simulation import himmelblau, branin


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
    n_samples: int,
    n_vars: int,
    lower_bound: List[float],
    upper_bound: List[float],
    seed: int,
    dataset_type: str,
    x_labels: List[str],
    y_labels: List[str],
) -> Dataset:
    """
    Uses LHS sampling to generate datasets.

    Parameters:
        n_samples (int): Number of samples to generate.
        n_vars (int): Number of dimensions.
        lower_bound (List[float]): Lower bound.
        upper_bound (List[float]): Upper bound.
        seed (int): Random seed for reproducibility.
        dataset_type (str): Dataset type.
        x_labels (List[str]): Labels for the dataset input.
        y_labels (List[str]): Labels for the dataset output.

    Returns:
        Dataset(x, y): dataset.
    """

    x = lhs_sampler(
        n_samples=n_samples,
        n_vars=n_vars,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        seed=seed,
    )
    y = np.array([branin(x)]).reshape(-1, 1)

    return Dataset(
        dataset_type=dataset_type,
        x_data=x,
        y_data=y,
        x_labels=x_labels,
        y_labels=y_labels,
    )


def generate_design_space_dataset(
    n_samples_per_region: int,
    delta: float,
    seed: int,
    dataset_type: str,
    x_labels: List[str],
    y_labels: List[str],
    maxima: Union[NDArray[np.float64], None] = None,
) -> Dataset:
    """
    Uses LHS sampling to generate dataset in the design space.

    Parameters:
        n_samples_per_region (int): Number of samples to generate per region.
        delta (float): indicates how large the region should be.
        seed (int): Random seed for reproducibility.
        dataset_type (str): Dataset type.
        x_labels (List[str]): Labels for the dataset input.
        y_labels (List[str]): Labels for the dataset output.
        maxima (Union[NDArray[np.float64], None]): maxima values of the Himmelblau function.

    Returns:
        Dataset(x_data, y_data): dataset in the design space.
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

    x_all, y_all = [], []

    for m in maxima:
        lower_bound = m - delta
        upper_bound = m + delta

        dataset = generate_data(
            n_samples=n_samples_per_region,
            n_vars=2,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            seed=seed,
            dataset_type=dataset_type,
            x_labels=x_labels,
            y_labels=y_labels,
        )

        x_all.append(dataset.x_data)
        y_all.append(dataset.y_data)

    return Dataset(
        dataset_type=dataset_type,
        x_data=np.vstack(x_all),
        y_data=np.vstack(y_all),
        x_labels=x_labels,
        y_labels=y_labels,
    )
