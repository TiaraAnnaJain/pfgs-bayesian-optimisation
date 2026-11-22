"""Unit tests for acquisition functions."""

import numpy as np

from pfgs_bo.bo.acquisition import compute_ei, compute_ucb
from pfgs_bo.data.data_models import Dataset


def test_ucb():
    """Testing UCB acquisition function."""
    x_data = np.array([[0], [1]])
    x_label = ["x"]
    y_label = ["y"]
    y_data = np.array([0.5, 0.5])
    y_errors = np.array([0.1, 1.0])

    preds = Dataset(
        dataset_type="predictions",
        x_data=x_data,
        x_labels=x_label,
        y_labels=y_label,
        y_data=y_data,
        y_errors=y_errors,
    )
    ucb = compute_ucb(preds, beta=1.0)

    assert ucb[1] > ucb[0]


def test_ei():
    """Testing EI acquisition function."""
    x_data = np.array([[0], [1]])
    x_label = ["x"]
    y_label = ["y"]
    y_data = np.array([0.5, 0.8])
    y_pred = np.array([1.0, 2.0])
    y_errors = np.array([0.5, 0.5])

    preds = Dataset(
        dataset_type="predictions",
        x_data=x_data,
        x_labels=x_label,
        y_labels=y_label,
        y_data=y_pred,
        y_errors=y_errors,
    )
    train = Dataset(
        dataset_type="train",
        x_data=x_data,
        y_data=y_data,
        x_labels=x_label,
        y_labels=y_label,
    )

    ei = compute_ei(preds, train)

    assert ei[1] > ei[0]
