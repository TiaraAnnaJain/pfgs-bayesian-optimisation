"""Unit tests for data processing and standardisation."""

import numpy as np
import pytest
from src.data.data_processing import himmelblau, generate_data
from src.data.data_models import Dataset


def test_himmelblau():
    """Test Himmelblau function by testing values at maxima."""
    X_maxima = np.array([[3, 2]])
    y = himmelblau(X_maxima).squeeze()
    assert float(y) == float(np.exp(8.01))


def test_generate_data():
    """Test data generation and LHS sampling."""
    train, test, design_space, _, _ = generate_data()
    assert train.x.shape == (5, 2)
    assert test.x.shape == (100, 2)
    assert design_space.x.shape == (100, 2)
    assert train.y.shape == (5, 1)
    assert test.y.shape == (100, 1)
    assert design_space.y.shape == (100, 1)

    assert train.x.min() >= -5
    assert train.x.max() <= 5
    assert test.x.min() >= -5
    assert test.x.max() <= 5
    assert design_space.x.min() >= -6
    assert design_space.x.max() <= 6


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
