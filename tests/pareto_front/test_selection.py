"""Tests for optimal point calculator."""

import numpy as np
import pytest

from pfgs_bo.data.data_models import Dataset
from pfgs_bo.pareto_front.selection import ParetoPointSelector


def test_closest_to_utopia_point():
    pareto_front = Dataset(
        dataset_type="pareto_front",
        x_data=np.array([[2, 0.2], [0.8, 0.8], [0.5, 0.5], [0.9, 0.8]]),
        x_labels=["x1", "x2"],
        y_labels=["y1"],
        y_data=np.array([[2], [8], [5], [9]]),
        y_errors=np.array([[2], [8], [5], [8]]),
    )
    utopia_point = np.array([1.0, 1.0])

    calculator = ParetoPointSelector(
        pareto_front=pareto_front, utopia_point=utopia_point
    )
    _, filtered_front, _, _ = calculator.discard_areas(
        mean_discard_threshold=np.array([0]), std_dev_discard_threshold=np.array([0])
    )
    _, closest_idx, closest_point, _ = calculator.closest_points(filtered_front)

    assert np.allclose(closest_point, [[9, 8]])
    assert closest_idx == 3


def test_discard_areas():
    pareto_front = Dataset(
        dataset_type="pareto_front",
        x_data=np.array([[2, 0.2], [0.8, 0.8], [0.5, 0.5], [0.9, 0.8]]),
        x_labels=["x1", "x2"],
        y_labels=["y1"],
        y_data=np.array([[2], [8], [5], [9]]),
        y_errors=np.array([[2], [8], [5], [8]]),
    )
    utopia_point = np.array([0, 0])

    mean_discard_threshold = np.array([3])
    std_dev_discard_threshold = np.array([3])

    calculator = ParetoPointSelector(
        pareto_front=pareto_front, utopia_point=utopia_point
    )
    _, filtered_front, _, _ = calculator.discard_areas(
        mean_discard_threshold=mean_discard_threshold,
        std_dev_discard_threshold=std_dev_discard_threshold,
    )

    _, _, closest_point, _ = calculator.closest_points(filtered_front)

    assert np.allclose(closest_point, [[5, 5]])


def test_rejects_normalised_pareto_front():
    pareto_front = Dataset(
        dataset_type="pareto_front",
        x_data=np.array([[0.1, 0.2]]),
        x_labels=["x1", "x2"],
        y_labels=["y1"],
        y_data=np.array([[1.0]]),
        y_errors=np.array([[0.5]]),
        normalised=True,
    )

    with pytest.raises(ValueError, match="must not be normalised"):
        ParetoPointSelector(pareto_front=pareto_front, utopia_point=np.array([1, 1]))
