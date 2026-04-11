"""Unit tests for data processing."""

import numpy as np

from src.data.data_processing import himmelblau, generate_data


def test_himmelblau():
    """Test Himmelblau function by testing values at maxima."""
    x_maxima = np.array([[3, 2]])
    y = himmelblau(x_maxima).squeeze()
    assert float(y) == float(np.exp(8.01))


def test_generate_data():
    """Test data generation and LHS sampling."""
    train, test, design_space = generate_data()
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
