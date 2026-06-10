"""Data models for all GP datasets including recipes, experiments and predictions."""

from typing import Optional, Dict, List, Union, Self
import logging as log
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray


@dataclass
class Dataset:
    """
    Dataset for the GP. Contains normalisation and de-normalisation functions.

    Parameters:
        dataset_type (str): type of dataset.
        x_labels (List[str]): labels for the dataset input.
        y_labels (List[str]): labels for the dataset output.
        x_data (Optional[NDArray[np.float64]]): input data.
        y_data (Optional[NDArray[np.float64]]): output data.
        y_errors (Optional[NDArray[np.float64]]): y error values from GP predictions.
        normalised (bool): normalisation flag to prevent double transformations.
        x_normaliser_min (Dict[str, float]): minimum value of x taken from bounds.
        x_normaliser_max (Dict[str, float]): maximum value of x taken from bounds.
        y_standardiser_mean (Dict[str, float]): mean value of y computed on
            training dataset outputs.
        y_standardiser_std (Dict[str, float]): standard deviation value of y computed
            on training dataset outputs.
    """

    dataset_type: str  # train, test, predictions, recipe, pareto_front

    x_labels: Union[List[str], None] = None
    y_labels: Union[List[str], None] = None

    x_data: Optional[NDArray[np.float64]] = None
    y_data: Optional[NDArray[np.float64]] = None
    y_errors: Optional[NDArray[np.float64]] = None

    normalised: bool = False

    x_normaliser_min: Dict[str, float] = field(default_factory=dict)
    x_normaliser_max: Dict[str, float] = field(default_factory=dict)
    y_standardiser_mean: Dict[str, float] = field(default_factory=dict)
    y_standardiser_std: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.check()

    def __neg__(self) -> Self:
        if self.dataset_type in ("train", "test"):
            raise ValueError("Cannot negate a train or test dataset.")

        return Dataset(
            dataset_type="predictions",
            x_labels=self.x_labels,
            y_labels=self.y_labels,
            x_data=self.x_data,
            y_data=-self.y_data,
            y_errors=-self.y_errors,
        )

    def check(self):
        """Runs basic sanity checks."""
        if self.dataset_type not in [
            "train",
            "test",
            "pareto_front",
            "recipe",
            "predictions",
        ]:
            raise ValueError(
                f"Dataset type {self.dataset_type} is not supported. Must be one of "
                f"['train', 'test', 'recipe', 'pareto_front', 'predictions']."
            )

        if self.x_labels is None:
            raise ValueError("x_labels must be provided.")

        if self.x_data is None:
            raise ValueError("Inputs must be provided.")

        if self.dataset_type in ("train", "test"):
            if self.y_data is None:
                raise ValueError("Outputs must be provided.")
            if self.y_labels is None:
                raise ValueError("y_labels must be provided.")

        if self.dataset_type == "recipe":
            if self.y_data is not None:
                raise ValueError("Recipe outputs should not be provided.")

        if self.dataset_type in ("predictions", "pareto_front"):
            if self.y_data is None or self.y_errors is None:
                raise ValueError("Mean and variance must be provided.")
            if self.y_labels is None:
                raise ValueError("y_labels must be provided.")

        if not isinstance(self.x_labels, list) or not all(
            isinstance(label, str) for label in self.x_labels
        ):
            raise ValueError(
                f"Labels {self.x_labels} is not supported. Labels must be a list of strings."
            )

        if len(self.x_labels) != self.x_data.shape[1]:
            raise ValueError(
                f"Number of x_labels must match number of columns in x_data. "
                f"\nNumber of labels provided: {len(self.x_labels)} "
                f"\nNumber of columns in x_data: {self.x_data.shape[1]}"
            )

    def add_normalisation_params(
        self, lower_bound: Dict[str, float], upper_bound: Dict[str, float]
    ) -> None:
        """
        Takes the lower and upper bounds of the investigation space and sets them as
        the normalisation parameters for x.

        Parameters:
            lower_bound (Dict[str, float]): Lower bounds of the investigation space, will be
                set as x_normaliser_min.
            upper_bound (Dict[str, float]): Upper bounds of the investigation space, will be
                set as x_normaliser_max.

        """
        if self.normalised:
            log.warning("Dataset already has normalisation parameters.")

        if set(lower_bound.keys()) != set(self.x_labels):
            raise ValueError("Lower bound labels do not match x_labels.")

        if set(upper_bound.keys()) != set(self.x_labels):
            raise ValueError("Upper bound labels do not match x_labels.")

        for label in self.x_labels:
            if lower_bound[label] >= upper_bound[label]:
                raise ValueError(
                    f"Lower bound must be less than upper bound for {label}."
                )

            self.x_normaliser_min[label] = lower_bound[label]
            self.x_normaliser_max[label] = upper_bound[label]

    def normalise(self):
        """
        Normalises (inputs) and standardises (outputs) the dataset. Fits and transforms the
        first dataset by computing the minimum and maximum of the inputs and mean and standard
        deviation of the outputs of the dataset. Transforms any later datasets by reusing the
        computed statistics.
        """
        if self.normalised:
            return

        if self.x_data is None or self.x_labels is None:
            raise ValueError(
                "Dataset normalisation requires x_data and x_labels to be provided."
            )

        self.x_data = np.asarray(self.x_data, dtype=np.float64)

        for i, label in enumerate(self.x_labels):
            if label not in self.x_normaliser_min or label not in self.x_normaliser_max:
                raise ValueError("Must provide both minimum and maximum values of x.")

            if self.x_normaliser_min[label] == self.x_normaliser_max[label]:
                raise ValueError(
                    "Minimum value == maximum value, cannot normalise dataset."
                )

            self.x_data[:, i] = (self.x_data[:, i] - self.x_normaliser_min[label]) / (
                self.x_normaliser_max[label] - self.x_normaliser_min[label]
            )

        if self.y_data is not None:
            self.y_data = np.asarray(self.y_data, dtype=np.float64)

            for i, label in enumerate(self.y_labels):
                if (
                    label not in self.y_standardiser_mean
                    or label not in self.y_standardiser_std
                ):
                    self.y_standardiser_mean[label] = self.y_data[:, i].mean()
                    self.y_standardiser_std[label] = self.y_data[:, i].std()

                    if self.y_standardiser_std[label] == 0:
                        raise ValueError(
                            "Standard deviation is 0, cannot normalise dataset."
                        )

                self.y_data[:, i] = (
                    self.y_data[:, i] - self.y_standardiser_mean[label]
                ) / self.y_standardiser_std[label]

                if self.y_errors is not None:
                    self.y_errors[:, i] = self.y_errors[:, i] / (
                        self.y_standardiser_std[label] ** 2
                    )

        self.normalised = True

    def de_normalise(self):
        """De-normalises the dataset using already computed statistics."""
        if not self.normalised:
            return

        if self.x_data is not None:
            self.x_data = np.asarray(self.x_data, dtype=np.float64)
            for i, label in enumerate(self.x_labels):
                self.x_data[:, i] = (
                        self.x_data[:, i]
                        * (self.x_normaliser_max[label] - self.x_normaliser_min[label])
                        + self.x_normaliser_min[label]
                )

        if self.y_data is not None:
            self.y_data = np.asarray(self.y_data, dtype=np.float64)

            if self.y_data.ndim == 1:
                self.y_data = self.y_data.reshape(-1, 1)

            for i, label in enumerate(self.y_labels):
                self.y_data[:, i] = (self.y_data[:, i] * self.y_standardiser_std[label]) + self.y_standardiser_mean[
                    label]

                if self.y_errors is not None:
                    if self.y_errors.ndim == 1:
                        self.y_errors = self.y_errors.reshape(-1, 1)
                    self.y_errors = np.asarray(self.y_errors, dtype=np.float64)
                    self.y_errors[:, i] = (
                            self.y_errors[:, i] * self.y_standardiser_std[label] ** 2
                    )

        self.normalised = False

    def get_stats(self) -> Dict[str, Dict]:
        """
        Returns already computed statistics to be used for transformations/de-normalisation.

        Returns:
            stats (Dict[str, NDArray[np.float64]]): computed statistics.
        """
        stats = {
            "x_min": self.x_normaliser_min,
            "x_max": self.x_normaliser_max,
            "y_mean": self.y_standardiser_mean,
            "y_std": self.y_standardiser_std,
        }

        return stats

    def set_stats(self, stats: Dict[str, Dict]):
        """
        Sets pre-computed statistics to be used for transformations/de-normalisation.

        Parameters:
            stats (Dict[str, NDArray[np.float64]]): computed statistics.
        """
        self.x_normaliser_min = stats["x_min"]
        self.x_normaliser_max = stats["x_max"]
        self.y_standardiser_mean = stats["y_mean"]
        self.y_standardiser_std = stats["y_std"]

    def copy(self) -> Self:
        """Creates a deepcopy of a dataset."""
        return deepcopy(self)
