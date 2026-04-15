"""Data models for all GP datasets including recipes, experiments and predictions."""

from typing import Optional, Dict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Dataset:
    """
    Dataset for the GP. Contains standardisation and de-standardisation functions.
    """

    x_data: Optional[NDArray[np.float64]] = None
    y_data: Optional[NDArray[np.float64]] = None
    y_errors: Optional[NDArray[np.float64]] = None

    standardised: bool = False

    x_standardiser_mean: NDArray[np.float64] = None
    x_standardiser_std: NDArray[np.float64] = None
    y_standardiser_mean: NDArray[np.float64] = None
    y_standardiser_std: NDArray[np.float64] = None

    def standardise(self):
        """
        Standardises the dataset. Fits and transforms the first dataset by computing the
        mean and standard deviation. Transforms any later datasets by reusing the
        computed statistics.
        """
        if self.standardised:
            return

        if self.x_data is None:
            raise ValueError("Dataset standardisation requires x to be provided.")

        self.x_data = np.asarray(self.x_data, dtype=np.float64)
        self.y_data = np.asarray(self.y_data, dtype=np.float64)

        if self.x_standardiser_mean is None or self.x_standardiser_std is None:
            self.x_standardiser_mean = self.x_data.mean(axis=0)
            self.x_standardiser_std = self.x_data.std(axis=0)

            if np.any(self.x_standardiser_std == 0):
                raise ValueError("X standard deviation is 0, cannot standardize.")

        if self.y_standardiser_mean is None or self.y_standardiser_std is None:
            self.y_standardiser_mean = self.y_data.mean(axis=0)
            self.y_standardiser_std = self.y_data.std(axis=0)

            if np.any(self.y_standardiser_std == 0):
                raise ValueError("Y standard deviation is 0, cannot standardize.")

        self.x_data = (self.x_data - self.x_standardiser_mean) / self.x_standardiser_std
        self.y_data = (self.y_data - self.y_standardiser_mean) / self.y_standardiser_std

        self.standardised = True

    def de_standardise(self):
        """De-standardises the dataset using already computed statistics."""
        if not self.standardised:
            return

        if self.x_data is not None:
            self.x_data = (
                self.x_data * self.x_standardiser_std
            ) + self.x_standardiser_mean

        self.y_data = self.y_data * self.y_standardiser_std + self.y_standardiser_mean

        if self.y_errors is not None:
            self.y_errors = np.asarray(self.y_errors, dtype=np.float64)

            self.y_errors = self.y_errors * self.y_standardiser_std**2

        self.standardised = False

    def get_stats(self) -> Dict[str, NDArray[np.float64]]:
        """
        Returns already computed statistics to be used for transformations/de-standardisation.

        Returns:
            stats (Dict[str, NDArray[np.float64]]): computed statistics.
        """
        stats = {
            "x_mean": self.x_standardiser_mean,
            "x_std": self.x_standardiser_std,
            "y_mean": self.y_standardiser_mean,
            "y_std": self.y_standardiser_std,
        }

        return stats

    def set_stats(self, stats: Dict[str, NDArray[np.float64]]):
        """
        Sets pre-computed statistics to be used for transformations/de-standardisation.

        Parameters:
            stats (Dict[str, NDArray[np.float64]]): computed statistics.
        """
        self.x_standardiser_mean = stats["x_mean"]
        self.x_standardiser_std = stats["x_std"]
        self.y_standardiser_mean = stats["y_mean"]
        self.y_standardiser_std = stats["y_std"]
