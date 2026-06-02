"""Data models for all GP datasets including recipes, experiments and predictions."""

from typing import Optional, Dict, List, Union
from dataclasses import dataclass, field

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
        x_normaliser_min (Dict[str, float]): minimum value of x computed on
            training dataset inputs.
        x_normaliser_max (Dict[str, float]): maximum value of x computed
            on training dataset inputs.
        y_normaliser_min (Dict[str, float]): minimum value of y computed on
            training dataset outputs.
        y_normaliser_max (Dict[str, float]): maximum value of y computed
            on training dataset outputs.
    """

    dataset_type: str = None  # train, test, predictions

    x_labels: Union[List[str], None] = None
    y_labels: Union[List[str], None] = None

    x_data: Optional[NDArray[np.float64]] = None
    y_data: Optional[NDArray[np.float64]] = None
    y_errors: Optional[NDArray[np.float64]] = None

    normalised: bool = False

    x_normaliser_min: Dict[str, float] = field(default_factory=dict)
    x_normaliser_max: Dict[str, float] = field(default_factory=dict)
    y_normaliser_min: Dict[str, float] = field(default_factory=dict)
    y_normaliser_max: Dict[str, float] = field(default_factory=dict)

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

    def normalise(self):
        """
        Normalises the dataset. Fits and transforms the first dataset by computing the
        minimum and maximum of the dataset. Transforms any later datasets by reusing the
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
                self.x_normaliser_min[label] = self.x_data[:, i].min()
                self.x_normaliser_max[label] = self.x_data[:, i].max()

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
                    label not in self.y_normaliser_min
                    or label not in self.y_normaliser_max
                ):
                    self.y_normaliser_min[label] = self.y_data[:, i].min()
                    self.y_normaliser_max[label] = self.y_data[:, i].max()

                    if self.y_normaliser_min[label] == self.y_normaliser_max[label]:
                        raise ValueError(
                            "Minimum value == maximum value, cannot normalise dataset."
                        )

                self.y_data[:, i] = (
                    2
                    * (self.y_data[:, i] - self.y_normaliser_min[label])
                    / (self.y_normaliser_max[label] - self.y_normaliser_min[label])
                    - 1
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

        self.y_data = np.asarray(self.y_data, dtype=np.float64)
        for i, label in enumerate(self.y_labels):
            self.y_data[:, i] = ((self.y_data[:, i] + 1) / 2) * (
                self.y_normaliser_max[label] - self.y_normaliser_min[label]
            ) + self.y_normaliser_min[label]

            if self.y_errors is not None:
                self.y_errors = np.asarray(self.y_errors, dtype=np.float64)
                self.y_errors[:, i] = (
                    self.y_errors[:, i]
                    * (
                        (self.y_normaliser_max[label] - self.y_normaliser_min[label])
                        / 2
                    )
                    ** 2
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
            "y_min": self.y_normaliser_min,
            "y_max": self.y_normaliser_max,
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
        self.y_normaliser_min = stats["y_min"]
        self.y_normaliser_max = stats["y_max"]
