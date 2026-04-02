"""Data models for all GP datasets."""

from dataclasses import dataclass
from typing import Tuple, Optional, Union
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
        Standardises x and y.

        Returns:
            self.x (NDArray[np.float64]): standardised x.
            self.y (NDArray[np.float64]): standardised y.
        """
        if self.standardised:
            return

        self.x_standardiser_mean = self.x.mean(axis=0)
        self.x_standardiser_std = self.x.std(axis=0)
        self.y_standardiser_mean = self.y.mean(axis=0)
        self.y_standardiser_std = self.y.std(axis=0)

        if np.any(self.x_standardiser_std == 0):
            raise ValueError("X standard deviation has zero(s), cannot standardize.")
        if np.any(self.y_standardiser_std == 0):
            raise ValueError("Y standard deviation has zero(s), cannot standardize.")

        self.x = (self.x - self.x_standardiser_mean) / self.x_standardiser_std
        self.y = (self.y - self.y_standardiser_mean) / self.y_standardiser_std

        self.standardised = True

    def transform(
        self, x_new: NDArray[np.float64], y_new: Optional[NDArray[np.float64]]
    ) -> Union[
        Tuple[NDArray[np.float64], None],
        Tuple[NDArray[np.float64], NDArray[np.float64]],
    ]:
        """
        Transforms x and y.

        Parameters:
            x_new (NDArray[np.float64]): x to be standardised.
            y_new (Optional[NDArray[np.float64]]): y to be standardised (optional)

        Returns:
            x_new (NDArray[np.float64]): transformed x.
            y_new (NDArray[np.float64]) | None: transformed y if passed, None otherwise.
        """
        if not self.standardised:
            raise ValueError("Training data is not standardised.")

        x_new = (x_new - self.x_standardiser_mean) / self.x_standardiser_std
        if y_new is None:
            return x_new, None

        y_new = (y_new - self.y_standardiser_mean) / self.y_standardiser_std
        return x_new, y_new

    def transform_inverse(
        self, x_new: NDArray[np.float64], y_new: Optional[NDArray[np.float64]]
    ) -> Union[
        Tuple[NDArray[np.float64], None],
        Tuple[NDArray[np.float64], NDArray[np.float64]],
    ]:
        """
        Inverses transformed x.

        Parameters:
            x_new (NDArray[np.float64]): transformed x.
            y_new (Optional[NDArray[np.float64]]): transformed y (optional).

        Returns:
            x_new (NDArray[np.float64]): de-standardised X.
            y_new (NDArray[np.float64]) | None: de-standardised y if passed, None otherwise.
        """
        if not self.standardised:
            raise ValueError("Training data is not standardised.")

        x_new = (x_new + self.x_standardiser_mean) * self.x_standardiser_std
        if y_new is None:
            return x_new, None

        y_new = (y_new + self.y_standardiser_mean) * self.y_standardiser_std
        return x_new, y_new

    def de_standardize(
        self, predictions_standardised: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        De-standardises GP predictions.

        Parameters:
            predictions_standardised (NDArray[np.float64]): standardised predictions.

        Returns:
            predictions (NDArray[np.float64]): de-standardised predictions.
        """
        if not self.standardised:
            raise ValueError("Training data is not standardised.")

        mean = predictions_standardised[:, 0]
        var = predictions_standardised[:, 1]

        mean = mean * self.y_standardiser_std + self.y_standardiser_mean
        var = var * (self.y_standardiser_std**2)

        predictions = np.column_stack((mean, var))
        return predictions
