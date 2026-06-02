"""Data models for all GP datasets including recipes, experiments and predictions."""

from typing import Optional, Dict, List
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Dataset:
    """
    Dataset for the GP. Contains standardisation and de-standardisation functions.

    Parameters:
        dataset_type (str): type of dataset.
        x_labels (List[str]): labels for the dataset input.
        y_labels (List[str]): labels for the dataset output.
        x_data (Optional[NDArray[np.float64]]): input data.
        y_data (Optional[NDArray[np.float64]]): output data.
        y_errors (Optional[NDArray[np.float64]]): y error values from GP predictions.
        standardised (bool): standardisation flag to prevent double transformations.
        x_standardiser_mean (Dict[str, float]): mean value of x computed on
            training dataset inputs.
        x_standardiser_std (Dict[str, float]): standard deviation of x computed
            on training dataset inputs.
        y_standardiser_mean (Dict[str, float]): mean value of y computed on
            training dataset outputs.
        y_standardiser_std (Dict[str, float]): standard deviation of y computed
            on training dataset outputs.
    """

    dataset_type: str = None  # train, test, predictions

    x_labels: Optional[List[str]] = None
    y_labels: Optional[List[str]] = None

    x_data: Optional[NDArray[np.float64]] = None
    y_data: Optional[NDArray[np.float64]] = None
    y_errors: Optional[NDArray[np.float64]] = None

    standardised: bool = False

    x_standardiser_mean: Dict[str, float] = field(default_factory=dict)
    x_standardiser_std: Dict[str, float] = field(default_factory=dict)
    y_standardiser_mean: Dict[str, float] = field(default_factory=dict)
    y_standardiser_std: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.dataset_type not in ["train", "test", "predictions"]:
            raise ValueError(
                f"Dataset type {self.dataset_type} is not supported. Must be one of "
                f"['train', 'test', 'predictions']."
            )

        if self.dataset_type in ("train", "test"):
            if self.x_data is None or self.y_data is None:
                raise ValueError("Inputs and outputs must be provided.")

        if self.dataset_type == "predictions":
            if self.y_data is None or self.y_errors is None:
                raise ValueError("Mean and variance must be provided.")

        if self.x_labels is not None:
            if not isinstance(self.x_labels, list) or not all(
                isinstance(label, str) for label in self.x_labels
            ):
                raise ValueError(
                    f"Labels {self.x_labels} is not supported. Labels must be a list of strings."
                )

        if self.x_labels is not None and self.x_data is not None:
            if len(self.x_labels) != self.x_data.shape[1]:
                raise ValueError(
                    f"Number of x_labels must match number of columns in x_data. "
                    f"\nNumber of labels provided: {len(self.x_labels)} "
                    f"\nNumber of columns in x_data: {self.x_data.shape[1]}"
                )

    def __neg__(self) -> "Dataset":
        if self.dataset_type in ("train", "test"):
            raise ValueError("Cannot negate a train or test dataset.")

        return Dataset(
            dataset_type="predictions",
            y_data=-self.y_data,
            y_errors=-self.y_errors,
            y_labels=self.y_labels,
        )

    def standardise(self):
        """
        Standardises the dataset. Fits and transforms the first dataset by computing the
        mean and standard deviation. Transforms any later datasets by reusing the
        computed statistics.
        """
        if self.standardised:
            return

        if self.x_data is None or self.x_labels is None:
            raise ValueError(
                "Dataset standardisation requires x_data and x_labels to be provided."
            )

        self.x_data = np.asarray(self.x_data, dtype=np.float64)

        for i, label in enumerate(self.x_labels):
            if (
                label not in self.x_standardiser_mean
                or label not in self.x_standardiser_std
            ):
                self.x_standardiser_mean[label] = self.x_data[:, i].mean()
                self.x_standardiser_std[label] = self.x_data[:, i].std()

                if self.x_standardiser_std[label] == 0:
                    raise ValueError(
                        f"Standard deviation for {label} is 0, cannot standardise."
                    )

            self.x_data[:, i] = (
                self.x_data[:, i] - self.x_standardiser_mean[label]
            ) / self.x_standardiser_std[label]

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
                            f"Standard deviation for {label} is 0, cannot standardise."
                        )

                self.y_data[:, i] = (
                    self.y_data[:, i] - self.y_standardiser_mean[label]
                ) / self.y_standardiser_std[label]

        self.standardised = True

    def de_standardise(self):
        """De-standardises the dataset using already computed statistics."""
        if not self.standardised:
            return

        if self.x_data is not None:
            self.x_data = np.asarray(self.x_data, dtype=np.float64)
            for i, label in enumerate(self.x_labels):
                self.x_data[:, i] = (
                    self.x_data[:, i] * self.x_standardiser_std[label]
                ) + self.x_standardiser_mean[label]

        self.y_data = np.asarray(self.y_data, dtype=np.float64)
        for i, label in enumerate(self.y_labels):
            self.y_data[:, i] = (
                self.y_data[:, i] * self.y_standardiser_std[label]
            ) + self.y_standardiser_mean[label]

            if self.y_errors is not None:
                self.y_errors = np.asarray(self.y_errors, dtype=np.float64)
                self.y_errors[:, i] = (
                    self.y_errors[:, i] * self.y_standardiser_std[label] ** 2
                )

        self.standardised = False

    def get_stats(self) -> Dict[str, Dict]:
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

    def set_stats(self, stats: Dict[str, Dict]):
        """
        Sets pre-computed statistics to be used for transformations/de-standardisation.

        Parameters:
            stats (Dict[str, NDArray[np.float64]]): computed statistics.
        """
        self.x_standardiser_mean = stats["x_mean"]
        self.x_standardiser_std = stats["x_std"]
        self.y_standardiser_mean = stats["y_mean"]
        self.y_standardiser_std = stats["y_std"]
