"""Calculations for optimal point on pareto front."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

from data.data_models import Dataset


class ParetoPointSelector:
    """
    Calculates the optimal point on Pareto front using the Euclidean distance from the
    defined utopia point.

    Parameters:
        pareto_front (Dataset): unscaled, de-normalised pareto front points.
        utopia_point (NDArray[np.float64]): utopia point as reference point.
    """

    def __init__(
        self,
        pareto_front: Dataset,
        utopia_point: NDArray[np.float64],
    ) -> None:
        self.pareto_front = pareto_front
        self.utopia_point = utopia_point

    def __post__init(self):
        """Validations to make sure pareto front is de-normalised."""
        if self.pareto_front.normalised:
            raise ValueError("Pareto front must not be normalised.")

        if self.pareto_front.y_data is None or self.pareto_front.y_errors is None:
            raise ValueError(
                "Pareto front must contain both mean and standard deviation."
            )

    def discard_areas(
        self,
        mean_discard_threshold: NDArray[np.float64],
        std_dev_discard_threshold: NDArray[np.float64],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        bool,
    ]:
        """
        Discards points on Pareto front with mean and standard deviation values lower than the
        threshold input. If no point survives, the optimisation loop breaks.

        Parameters:
            mean_discard_threshold (NDArray[np.float64]): threshold below which mean
            values will be discarded.
            std_dev_discard_threshold (NDArray[np.float64]): threshold below which
                standard deviation values will be discarded.

        Returns:
            mask (NDArray[np.float64]): mask of points discarded.
            filtered_pareto_front (NDArray[np.float64]): filtered pareto front
                points.
            stop_optimisation (bool): flag to stop optimisation.
        """
        pareto_front_points = np.column_stack(
            (self.pareto_front.y_data, self.pareto_front.y_errors)
        )
        discard_thresholds = np.hstack(
            [mean_discard_threshold, std_dev_discard_threshold]
        )

        if discard_thresholds.shape[0] != pareto_front_points.shape[1]:
            raise ValueError(
                "Number of mean + standard deviation thresholds must match Pareto objectives."
            )

        stop_optimisation = False
        mask = (pareto_front_points >= discard_thresholds).all(axis=1)

        if not mask.any():
            stop_optimisation = True

        filtered_pareto_front = pareto_front_points[mask]

        return (
            mask,
            filtered_pareto_front,
            discard_thresholds,
            stop_optimisation,
        )

    def distance_calculator(
        self, scaled_filtered_pareto_front: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Euclidean distance calculator between points on pareto front and utopia point.

        Parameters:
            scaled_filtered_pareto_front (NDArray[np.float64]): filtered scaled pareto
                front points.

        Returns:
            distance (NDArray[np.float64]): vector of Euclidean distances.
        """
        distance = np.linalg.norm(
            scaled_filtered_pareto_front - self.utopia_point, axis=1
        )
        return distance

    def closest_points(
        self,
        filtered_pareto_front: NDArray[np.float64],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Finds the closest point on Pareto front by minimising Euclidean distance from
        utopia point.

        Parameters:
            filtered_pareto_front (NDArray[np.float64]): filtered pareto front.

        Returns:
            closest_point (NDArray[np.float64]): closest point on scaled Pareto front.
            closest_idx (NDArray[np.float64]): indices of closest point on Pareto front.
            optimal_point (NDArray[np.float64]): closest point on unscaled Pareto front.
            scaled_pareto_front (NDArray[np.float64]): scaled, unfiltered Pareto front.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_filtered_pareto_front = scaler.fit_transform(filtered_pareto_front)
        scaled_pareto_front = scaler.fit_transform(
            np.column_stack((self.pareto_front.y_data, self.pareto_front.y_errors))
        )

        distance = self.distance_calculator(scaled_filtered_pareto_front)
        min_distance = distance.min()

        closest_idx = np.where(distance == min_distance)[0][0]

        optimal_point = filtered_pareto_front[closest_idx]
        optimal_point = np.array(optimal_point).reshape(-1, 2)

        closest_point = scaled_filtered_pareto_front[closest_idx]
        closest_point = np.array(closest_point).reshape(-1, 2)

        return closest_point, closest_idx, optimal_point, scaled_pareto_front
