from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src.data.data_models import Dataset


@dataclass(frozen=True)
class InvestigationSpace:
    """
    Dataset for investigation space for optimisers. Contains normalisation functions to
    normalise investigation space.

    Parameters:
        train_dataset (Dataset): train dataset from which normalisation statistics are
            used to compute the normalised space.
        lower_bounds (NDArray[np.float64]): not normalised lower bounds of the
            investigation space.
        upper_bounds (NDArray[np.float64]): not normalised upper bounds of the
            investigation space.
    """

    train_dataset: Dataset
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]

    def __post_init__(self):
        if self.train_dataset.x_labels is None:
            raise ValueError("Training dataset must have x_labels.")

        if not isinstance(self.lower_bounds, np.ndarray):
            raise TypeError("lower_bounds must be a NumPy array.")

        if not isinstance(self.upper_bounds, np.ndarray):
            raise TypeError("upper_bounds must be a NumPy array.")

        if self.lower_bounds.shape != self.upper_bounds.shape:
            raise ValueError("lower_bounds and upper_bounds must have the same shape.")

        if self.lower_bounds.ndim != 1:
            raise ValueError("Bounds must be 1D arrays.")

        if len(self.lower_bounds) != len(self.train_dataset.x_labels):
            raise ValueError(
                "Number of bounds must match number of x_labels in train_dataset."
            )

        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError(
                "Each lower bound must be less than or equal to upper bound."
            )

    @property
    def bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.lower_bounds, self.upper_bounds

    @property
    def n_var(self) -> int:
        return len(self.train_dataset.x_labels)

    def normalised_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Normalises the investigation space using training data statistics.

        Returns:
            lower_normalised (NDArray[np.float64]): normalised lower bounds of the investigation space.
            upper_normalised (NDArray[np.float64]): normalised upper bounds of the investigation space.
        """
        lower_normalised = []
        upper_normalised = []

        for i, label in enumerate(self.train_dataset.x_labels):
            x_min = self.train_dataset.x_normaliser_min[label]
            x_max = self.train_dataset.x_normaliser_max[label]

            lower_normalised.append((self.lower_bounds[i] - x_min) / (x_max - x_min))

            upper_normalised.append((self.upper_bounds[i] - x_min) / (x_max - x_min))

        return np.array(lower_normalised), np.array(upper_normalised)
