"""Unit tests for data processing and standardisation."""

import numpy as np
from src.gp_model.data_processing import (
    himmelblau,
    generate_data,
    DataStandardisation,
    SampleData,
    to_tensor,
)


def test_himmelblau():
    """Test Himmelblau function by testing values at maxima."""
    X_maxima = np.array([[3, 2]])
    y = himmelblau(X_maxima).squeeze()
    assert float(y) == float(np.exp(8.01))


def test_generate_data():
    """Test data generation and LHS sampling."""
    train, test, design_space, _, _ = generate_data()
    assert train.inputs.shape == (5, 2)
    assert test.inputs.shape == (100, 2)
    assert design_space.inputs.shape == (100, 2)
    assert train.outputs.shape == (5, 1)
    assert test.outputs.shape == (100, 1)
    assert design_space.outputs.shape == (100, 1)

    assert train.inputs.min() >= -5
    assert train.inputs.max() <= 5
    assert test.inputs.min() >= -5
    assert test.inputs.max() <= 5
    assert design_space.inputs.min() >= -6
    assert design_space.inputs.max() <= 6


def test_standardize():
    """Test standardization function."""
    data = SampleData(
        inputs=np.array([[1.0, 2.0], [3.0, 4.0]]), outputs=np.array([[10.0], [20.0]])
    )

    x_mean = np.mean(data.inputs, axis=0)
    x_std = np.std(data.inputs, axis=0)

    y_mean = np.mean(data.outputs, axis=0)
    y_std = np.std(data.outputs, axis=0)

    std = DataStandardisation(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    x_std, y_std = std.standardize(data)

    assert np.allclose(x_std, [[-1, -1], [1, 1]])
    assert np.allclose(y_std, [[-1], [1]])


def test_destandardize():
    """Test destandardization function."""
    std = DataStandardisation(
        x_mean=None, x_std=None, y_mean=np.array([10.0]), y_std=np.array([2.0])
    )

    preds = np.array([[0.0, 1.0], [1.0, 4.0]])

    result = std.de_standardize(preds)

    expected = np.array([[10.0, 4.0], [12.0, 16.0]])

    assert np.allclose(result, expected)
