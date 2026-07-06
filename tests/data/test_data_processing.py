"""Unit tests for data processing."""

import numpy as np
import pytest

from data.data_processing import generate_data, generate_design_space_dataset
from data.simulation import himmelblau


def test_himmelblau():
    """Test Himmelblau function by testing values at maxima."""
    x_maxima = np.array([[3, 2]])
    y = himmelblau(x_maxima).squeeze()
    assert float(y) == float(np.exp(8.01))


def test_himmelblau_input_dimensions():
    """Test the input dimensions of the Himmelblau function."""
    x = np.array([[1], [2]])

    with pytest.raises(ValueError):
        himmelblau(x)


def test_generate_data():
    """Test data generation and LHS sampling."""
    train = generate_data(
        simulation=himmelblau,
        n_samples=5,
        n_vars=2,
        lower_bound=[-5, -5],
        upper_bound=[5, 5],
        seed=42,
        dataset_type="train",
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )
    assert train.x_data.shape == (5, 2)
    assert train.y_data.shape == (5, 1)
    assert train.x_data.min() >= -5
    assert train.x_data.max() <= 5

    test = generate_data(
        simulation=himmelblau,
        n_samples=100,
        n_vars=2,
        lower_bound=[-5, -5],
        upper_bound=[5, 5],
        seed=10,
        dataset_type="test",
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )
    assert test.x_data.shape == (100, 2)
    assert test.y_data.shape == (100, 1)
    assert test.x_data.min() >= -5
    assert test.x_data.max() <= 5

    design_space = generate_design_space_dataset(
        simulation=himmelblau,
        n_samples=100,
        n_vars=2,
        delta=1.5,
        seed=10,
        dataset_type="test",
        x_labels=["x1", "x2"],
        y_labels=["y"],
        maxima=np.array(
            [
                [3.0, 2.0],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ]
        ),
    )
    assert design_space.x_data.shape == (100, 2)
    assert design_space.y_data.shape == (100, 1)
    assert design_space.x_data.min() >= -6
    assert design_space.x_data.max() <= 6
