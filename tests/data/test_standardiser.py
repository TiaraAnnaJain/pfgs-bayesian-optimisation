"""Unit tests for data model and standardisation"""

import numpy as np

from src.data.data_models import Dataset


def test_standardise():
    """Test standardisation function to fit and transform."""
    train_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_y = np.array([[10.0], [20.0]])

    train_data = Dataset(
        dataset_type="train",
        x_data=train_x,
        y_data=train_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )
    train_data.standardise()

    train_x_standardised = train_data.x_data
    train_y_standardised = train_data.y_data

    assert np.allclose(train_x_standardised, [[-1, -1], [1, 1]])
    assert np.allclose(train_y_standardised, [[-1], [1]])


def test_transform():
    """Test standardisation function to transform using already computed stats."""
    train_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_y = np.array([[10.0], [20.0]])

    train_data = Dataset(
        dataset_type="train",
        x_data=train_x,
        y_data=train_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )
    train_data.standardise()
    standardisation_stats = train_data.get_stats()

    test_x = np.array([[1.0, 3.0]])
    test_y = np.array([[30]])

    test_data = Dataset(
        dataset_type="test",
        x_data=test_x,
        y_data=test_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )
    test_data.set_stats(standardisation_stats)
    test_data.standardise()

    test_x_standardised = test_data.x_data
    test_y_standardised = test_data.y_data

    assert np.allclose(test_x_standardised, [[-1, 0]])
    assert np.allclose(test_y_standardised, [[3]])


def test_de_standardise():
    """Test de-standardisation function."""
    train_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_y = np.array([[10.0], [20.0]])

    train_data = Dataset(
        dataset_type="train",
        x_data=train_x,
        y_data=train_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )
    train_data.standardise()
    standardisation_stats = train_data.get_stats()

    prediction_mean = np.array([[0.0], [1.0]])
    prediction_error = np.array([[1.0], [4.0]])

    predictions = Dataset(
        dataset_type="predictions",
        y_data=prediction_mean,
        y_errors=prediction_error,
        y_labels=["y"],
    )
    predictions.standardised = True

    predictions.set_stats(standardisation_stats)
    predictions.de_standardise()

    predictions_mean_de_standardised = predictions.y_data
    predictions_error_de_standardised = predictions.y_errors

    expected_mean = np.array([[15.0], [20.0]])
    expected_error = np.array([[25.0], [100.0]])

    assert np.allclose(predictions_mean_de_standardised, expected_mean)
    assert np.allclose(predictions_error_de_standardised, expected_error)


def test_double_standardisation():
    """Test double standardisation and transformation."""
    train_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_y = np.array([[10.0], [20.0]])

    train_data = Dataset(
        dataset_type="train",
        x_data=train_x,
        y_data=train_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )

    train_data.standardise()
    train_data.standardise()

    train_x_standardised = train_data.x_data
    train_y_standardised = train_data.y_data

    assert np.allclose(train_x_standardised, [[-1, -1], [1, 1]])
    assert np.allclose(train_y_standardised, [[-1], [1]])

    standardisation_stats = train_data.get_stats()

    test_x = np.array([[1.0, 3.0]])
    test_y = np.array([[30]])

    test_data = Dataset(
        dataset_type="test",
        x_data=test_x,
        y_data=test_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )
    test_data.set_stats(standardisation_stats)
    test_data.standardise()
    test_data.standardise()

    test_x_standardised = test_data.x_data
    test_y_standardised = test_data.y_data

    assert np.allclose(test_x_standardised, [[-1, 0]])
    assert np.allclose(test_y_standardised, [[3]])


def test_double_de_standardisation():
    """Test double de-standardisation."""
    train_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_y = np.array([[10.0], [20.0]])

    train_data = Dataset(
        dataset_type="train",
        x_data=train_x,
        y_data=train_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )
    train_data.standardise()
    standardisation_stats = train_data.get_stats()

    prediction_mean = np.array([[0.0], [1.0]])
    prediction_error = np.array([[1.0], [4.0]])

    predictions = Dataset(
        dataset_type="predictions",
        y_data=prediction_mean,
        y_errors=prediction_error,
        y_labels=["y"],
    )
    predictions.standardised = True

    predictions.set_stats(standardisation_stats)
    predictions.de_standardise()
    predictions.de_standardise()

    predictions_mean_de_standardised = predictions.y_data
    predictions_error_de_standardised = predictions.y_errors

    expected_mean = np.array([[15.0], [20.0]])
    expected_error = np.array([[25.0], [100.0]])

    assert np.allclose(predictions_mean_de_standardised, expected_mean)
    assert np.allclose(predictions_error_de_standardised, expected_error)
