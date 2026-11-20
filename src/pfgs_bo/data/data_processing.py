"""Generates datasets using LHS sampling."""

from typing import List, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from pfgs_bo.data.data_models import Dataset


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
    simulation: Callable,
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
        simulation (Callable): Simulation function.
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
    y = np.array([simulation(x)]).reshape(-1, 1)

    return Dataset(
        dataset_type=dataset_type,
        x_data=x,
        y_data=y,
        x_labels=x_labels,
        y_labels=y_labels,
    )


def generate_design_space_dataset(
    simulation: Callable,
    n_samples: int,
    n_vars: int,
    delta: float,
    seed: int,
    dataset_type: str,
    x_labels: List[str],
    y_labels: List[str],
    maxima: NDArray[np.float64],
) -> Dataset:
    """
    Uses LHS sampling to generate dataset in the design space.

    Parameters:
        simulation (Callable): Simulation function.
        n_samples (int): Number of samples to generate.
        n_vars (int): Number of dimensions.
        delta (float): indicates how large the region should be.
        seed (int): Random seed for reproducibility.
        dataset_type (str): Dataset type.
        x_labels (List[str]): Labels for the dataset input.
        y_labels (List[str]): Labels for the dataset output.
        maxima (NDArray[np.float64]): maxima values of the simulation function.

    Returns:
        Dataset(x_data, y_data): dataset in the design space.
    """
    x_all, y_all = [], []
    n_samples_per_region = n_samples // maxima.shape[0]

    for m in maxima:
        lower_bound = m - delta
        upper_bound = m + delta

        dataset = generate_data(
            simulation=simulation,
            n_samples=n_samples_per_region,
            n_vars=n_vars,
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
