"""Calculations for optimal point on pareto front."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class ParetoPointSelector:
    """
    Calculates the optimal point on Pareto front using the Euclidean distance from the
    defined utopia point.

    Parameters:
        scaled_pareto_front (NDArray[np.float64]): scaled pareto front points.
        unscaled_pareto_front (NDArray[np.float64]): unscaled pareto front points.
        utopia_point (NDArray[np.float64]): utopia point as reference point.
    """

    def __init__(
        self,
        scaled_pareto_front: NDArray[np.float64],
        unscaled_pareto_front: NDArray[np.float64],
        utopia_point: NDArray[np.float64],
    ) -> None:
        self.scaled_pareto_front = scaled_pareto_front
        self.unscaled_pareto_front = unscaled_pareto_front
        self.utopia_point = utopia_point

    def discard_areas(
        self,
        mean_discard_threshold: NDArray[np.float64],
        variance_discard_threshold: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Discards points on Pareto front with mean and variance values lower than the threshold input.

        Parameters:
            mean_discard_threshold (NDArray[np.float64]): threshold below which mean values
                will be discarded.
            variance_discard_threshold (NDArray[np.float64]): threshold below which variance
                values will be discarded.

        Returns:
            mask (NDArray[np.float64]): mask of points discarded.
            filtered_scaled_pareto_front (NDArray[np.float64]): filtered pareto front
                points.
            filtered_unscaled_pareto_front (NDArray[np.float64]): original pareto front
                points.
        """
        discard_thresholds = np.hstack(
            [mean_discard_threshold, variance_discard_threshold]
        )

        if discard_thresholds.shape[0] != self.scaled_pareto_front.shape[1]:
            raise ValueError(
                "Number of mean + variance thresholds must match Pareto objectives."
            )

        mask = (self.scaled_pareto_front >= discard_thresholds).all(axis=1)

        filtered_scaled_pareto_front = self.scaled_pareto_front[mask]
        filtered_unscaled_pareto_front = self.unscaled_pareto_front[mask]

        if filtered_unscaled_pareto_front.shape[0] == 0:
            mask = np.ones(self.scaled_pareto_front.shape[0], dtype=bool)
            filtered_scaled_pareto_front = self.scaled_pareto_front
            filtered_unscaled_pareto_front = self.unscaled_pareto_front

        return mask, filtered_scaled_pareto_front, filtered_unscaled_pareto_front

    def distance_calculator(
        self, filtered_scaled_pareto_front: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Euclidean distance calculator between points on pareto front and utopia point.

        Parameters:
            filtered_scaled_pareto_front (NDArray[np.float64]): filtered scaled pareto
                front points.

        Returns:
            distance (NDArray[np.float64]): vector of Euclidean distances.
        """
        distance = np.linalg.norm(
            filtered_scaled_pareto_front - self.utopia_point, axis=1
        )
        return distance

    def closest_points(
        self, scaled_pareto_front: NDArray[np.float64], unscaled_pareto_front: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Finds the closest point on Pareto front by minimising Euclidean distance from
        utopia point.

        Parameters:
            scaled_pareto_front (NDArray[np.float64]): filtered, scaled pareto front.
            unscaled_pareto_front (NDArray[np.float64]): filtered, unscaled pareto front

        Returns:
            closest_point (NDArray[np.float64]): closest point on scaled Pareto front.
            closest_idx (NDArray[np.float64]): indices of closest point on Pareto front.
            optimal_point (NDArray[np.float64]): closest point on unscaled Pareto front.
        """
        distance = self.distance_calculator(scaled_pareto_front)
        min_distance = distance.min()

        closest_idx = np.where(distance == min_distance)[0][0]

        optimal_point = unscaled_pareto_front[closest_idx]
        optimal_point = np.array(optimal_point).reshape(-1, 2)

        closest_point = scaled_pareto_front[closest_idx]
        closest_point = np.array(closest_point).reshape(-1, 2)

        return closest_point, closest_idx, optimal_point
