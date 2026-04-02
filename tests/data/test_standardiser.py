"""Unit tests for data model and standardisation."""

import numpy as np
import pytest
from src.data.data_models import Dataset


def test_standardize():
    """Test standardization function."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[10.0], [20.0]])

    data = Dataset(x=X, y=y)

    data.standardize()
    x_std = data.x
    y_std = data.y

    assert np.allclose(x_std, [[-1, -1], [1, 1]])
    assert np.allclose(y_std, [[-1], [1]])


def test_transform_without_standardize_raises():
    X = np.array([[1.0], [2.0]])
    y = np.array([[3.0], [4.0]])

    data = Dataset(x=X, y=y)

    with pytest.raises(ValueError):
        data.transform(X, y)


def test_destandardize():
    """Test destandardization function."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[10.0], [20.0]])
    preds = np.array([[0.0, 1.0], [1.0, 4.0]])

    data = Dataset(x=X, y=y)

    data.standardize()
    result = data.de_standardize(preds)

    expected = np.array([[15.0, 25.0], [20.0, 100.0]])

    assert np.allclose(result, expected)
