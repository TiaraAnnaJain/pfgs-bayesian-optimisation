from dataclasses import dataclass
import numpy as np
import torch
from scipy.stats import qmc


def himmelblau(x: np.ndarray) -> np.ndarray:
    """Calculates the modified Himmelblau function."""
    x_1, x_2 = x[:, 0], x[:, 1]
    y = ((x_1**2 + x_2 - 11) ** 2) + ((x_1 + x_2**2 - 7) ** 2) + x_1 - (2 * x_2)
    return np.exp(8 - 0.01 * y)


@dataclass
class SampleData:
    inputs: np.ndarray
    outputs: np.ndarray


def generate_data(
    n_train: int = 5,
    n_test: int = 100,
    seed_train: int = 42,
    seed_test: int = 10,
    n_per_region: int = 25,
    delta: int = 1.5,
    maxima: np.ndarray | None = None,
) -> tuple[SampleData, SampleData, SampleData, int, int]:
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
        maxima (np.ndarray | None): maxima values of the Himmelblau function.

    Returns:
        SampleData(x_train, y_train): training dataset
        SampleData(x_test, y_test): test dataset
        SampleData(x_test_design_space_set,yx_test_design_space_set): test dataset in the design space only.
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
        SampleData(x_train, y_train),
        SampleData(x_test, y_test),
        SampleData(x_test_design_space_set, y_test_design_space_set),
        seed_train,
        seed_test,
    )


class DataStandardisation:
    """
    Standardises training and test datasets. De-standardises predictions.

    Parameters:
        x_mean (np.ndarray): training dataset inputs mean
        x_std (np.ndarray): training dataset inputs standard deviation
        y_mean (np.ndarray): training dataset outputs mean
        y_std (np.ndarray): training dataset outputs standard deviation
    """

    def __init__(self, x_mean, x_std, y_mean, y_std):
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def standardize(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        """
        Standardises dataset.

        Parameters:
            dataset (SampleData): dataset.

        Returns:
            x_standardised (np.ndarray): standardised inputs.
            y_standardised (np.ndarray): standardised outputs.
        """
        x_standardised = (dataset.inputs - self.x_mean) / self.x_std
        y_standardised = (dataset.outputs - self.y_mean) / self.y_std
        return x_standardised, y_standardised

    def de_standardize(self, preds: np.ndarray) -> np.ndarray:
        """
        De-standardises predictions.

        Parameters:
            preds (np.ndarray): GP predictions.

        Returns:
            preds_destandardised (np.ndarray): De-standardised predictions.
        """
        preds_mean = preds[:, 0]
        mean_destandardized = preds_mean * self.y_std + self.y_mean

        preds_var = preds[:, 1]
        var_destandardized = preds_var * (self.y_std**2)

        preds_destandardised = np.column_stack(
            (mean_destandardized, var_destandardized)
        )
        return preds_destandardised


def to_tensor(
    x_standardised: np.ndarray, y_standardised: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    x_tensor = torch.tensor(x_standardised, dtype=torch.float64)
    y_tensor = torch.tensor(y_standardised, dtype=torch.float64)
    return x_tensor, y_tensor
