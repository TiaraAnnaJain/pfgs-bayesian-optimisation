"""Unit tests for RMSE computation."""

import numpy as np

from pfgs_bo.metrics import compute_rmse, compute_rrmse


def test_compute_rmse():
    y_true = np.array([1, 2, 3, 4, 5])
    pred_y = np.array([[1], [2], [3], [4], [5]])

    rmse = compute_rmse(pred_y, y_true)
    assert rmse == 0


def test_compute_rrmse():
    y_true = np.array([1, 2, 3, 4, 5])
    pred_y = np.array([[2], [3], [4], [5], [6]])

    rrmse = compute_rrmse(pred_y, y_true)
    assert rrmse == (1 / (np.std(y_true)))
