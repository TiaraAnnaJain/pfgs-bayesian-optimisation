"""Data models for all GP datasets."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class Dataset:
    X: NDArray[np.float64]
    y: NDArray[np.float64]

    standardised: bool = False

    x_mean: NDArray[np.float64] = None
    x_std: NDArray[np.float64] = None
    y_mean: NDArray[np.float64] = None
    y_std: NDArray[np.float64] = None

    def standardize(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Standardises X and y.

        Returns:
            self.X (NDArray[np.float64]): standardised X.
            self.y (NDArray[np.float64]): standardised y.
        """
        if self.standardised:
            return

        self.x_mean = self.X.mean(axis=0)
        self.x_std = self.X.std(axis=0)
        self.y_mean = self.y.mean(axis=0)
        self.y_std = self.y.std(axis=0)

        self.x_std[self.x_std == 0] = 1.0
        if self.y_std == 0:
            self.y_std = np.array(1.0)

        self.X = (self.X - self.x_mean) / self.x_std
        self.y = (self.y - self.y_mean) / self.y_std

        self.standardised = True

        return self.X, self.y

    def transform(
        self, X_new: NDArray[np.float64], y_new: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Transforms X and y.

        Parameters:
            X_new (NDArray[np.float64]): X to be standardised.
            y_new (NDArray[np.float64]): y to be standardised.

        Returns:
            X_new (NDArray[np.float64]): transformed X.
            y_new (NDArray[np.float64]): transformed y.
        """
        if not self.standardised:
            raise ValueError("Training data is not standardised.")

        X_new = (X_new - self.x_mean) / self.x_std
        y_new = (y_new - self.y_mean) / self.y_std
        return X_new, y_new

    def transform_inverse(self, X_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Inverses transformed X.

        Parameters:
            X_new (NDArray[np.float64]): transformed X.

        Returns:
            X_new (NDArray[np.float64]): de-standardised X.
        """
        if not self.standardised:
            raise ValueError("Training data is not standardised.")

        X_new = (X_new + self.x_mean) * self.x_std
        return X_new

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

        mean = mean * self.y_std + self.y_mean
        var = var * (self.y_std**2)

        preds = np.column_stack((mean, var))
        return preds
