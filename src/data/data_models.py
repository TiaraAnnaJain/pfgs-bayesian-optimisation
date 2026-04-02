"""Data models for all GP datasets."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class Dataset:
    x: NDArray[np.float64]
    y: NDArray[np.float64]

    standardised: bool = False

    x_standardiser_mean: NDArray[np.float64] = None
    x_standardiser_std: NDArray[np.float64] = None
    y_standardiser_mean: NDArray[np.float64] = None
    y_standardiser_std: NDArray[np.float64] = None

    def standardize(self):
        """
        Standardises X and y.

        Returns:
            self.X (NDArray[np.float64]): standardised X.
            self.y (NDArray[np.float64]): standardised y.
        """
        if self.standardised:
            return

        self.x_standardiser_mean = self.x.mean(axis=0)
        self.x_standardiser_std = self.x.std(axis=0)
        self.y_standardiser_mean = self.y.mean(axis=0)
        self.y_standardiser_std = self.y.std(axis=0)

        self.x_standardiser_std[self.x_standardiser_std == 0] = 1.0
        if self.y_standardiser_std == 0:
            self.y_standardiser_std = np.array(1.0)

        self.x = (self.x - self.x_standardiser_mean) / self.x_standardiser_std
        self.y = (self.y - self.y_standardiser_mean) / self.y_standardiser_std

        self.standardised = True

    def transform(
        self, x_new: NDArray[np.float64], y_new: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Transforms X and y.

        Parameters:
            x_new (NDArray[np.float64]): X to be standardised.
            y_new (NDArray[np.float64]): y to be standardised.

        Returns:
            x_new (NDArray[np.float64]): transformed X.
            y_new (NDArray[np.float64]): transformed y.
        """
        if not self.standardised:
            raise ValueError("Training data is not standardised.")

        x_new = (x_new - self.x_standardiser_mean) / self.x_standardiser_std
        y_new = (y_new - self.y_standardiser_mean) / self.y_standardiser_std
        return x_new, y_new

    def transform_inverse(self, x_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Inverses transformed X.

        Parameters:
            x_new (NDArray[np.float64]): transformed X.

        Returns:
            x_new (NDArray[np.float64]): de-standardised X.
        """
        if not self.standardised:
            raise ValueError("Training data is not standardised.")

        x_new = (x_new + self.x_standardiser_mean) * self.x_standardiser_std
        return x_new

    def de_standardize(self, preds_std: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        De-standardises GP predictions.

        Parameters:
            preds_std (NDArray[np.float64]): standardised predictions.

        Returns:
            preds_std (NDArray[np.float64]): de-standardised predictions.
        """
        if not self.standardised:
            raise ValueError("Training data is not standardised.")

        mean = preds_std[:, 0]
        var = preds_std[:, 1]

        mean = mean * self.y_standardiser_std + self.y_standardiser_mean
        var = var * (self.y_standardiser_std**2)

        preds = np.column_stack((mean, var))
        return preds
