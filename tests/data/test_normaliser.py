"""Unit tests for data model and normalisation"""

import numpy as np
import pytest

from data.data_models import Dataset


def test_dataset_validation():
    """Test validation for different types of datasets."""
    train_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_y = np.array([[10.0], [20.0]])

    test_x = np.array([[1.0, 3.0]])

    recipe_y = np.array([[30]])

    prediction_mean = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError):
        Dataset(
            dataset_type="train",
            x_data=train_x,
            y_data=train_y,
        )

    with pytest.raises(ValueError):
        Dataset(
            dataset_type="test",
            x_data=test_x,
            x_labels=["x1", "x2"],
        )

    with pytest.raises(ValueError):
        Dataset(dataset_type="recipe", y_data=recipe_y)

    with pytest.raises(ValueError):
        Dataset(dataset_type="predictions", y_data=prediction_mean)


def test_normalise():
    """Test normalisation function to fit and transform."""
    train_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_y = np.array([[10.0], [20.0]])

    lower_bound = {"x1": 1.0, "x2": 2.0}
    upper_bound = {"x1": 3.0, "x2": 4.0}

    train_data = Dataset(
        dataset_type="train",
        x_data=train_x,
        y_data=train_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )

    train_data.add_normalisation_params(
        lower_bound=lower_bound, upper_bound=upper_bound
    )
    train_data.normalise()

    train_x_normalised = train_data.x_data
    train_y_normalised = train_data.y_data

    assert np.allclose(train_x_normalised, [[0, 0], [1, 1]])
    assert np.allclose(train_y_normalised, [[-1], [1]])


def test_transform():
    """Test normalisation function to transform using already computed stats."""
    train_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_y = np.array([[10.0], [20.0]])

    lower_bound = {"x1": 1.0, "x2": 2.0}
    upper_bound = {"x1": 3.0, "x2": 4.0}

    train_data = Dataset(
        dataset_type="train",
        x_data=train_x,
        y_data=train_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )

    train_data.add_normalisation_params(
        lower_bound=lower_bound, upper_bound=upper_bound
    )
    train_data.normalise()
    normalisation_stats = train_data.get_stats()

    test_x = np.array([[1.0, 3.0]])
    test_y = np.array([[30]])

    test_data = Dataset(
        dataset_type="test",
        x_data=test_x,
        y_data=test_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )
    test_data.set_stats(normalisation_stats)
    test_data.normalise()

    test_x_normalised = test_data.x_data
    test_y_normalised = test_data.y_data

    assert np.allclose(test_x_normalised, [[0, 0.5]])
    assert np.allclose(test_y_normalised, [[3]])


def test_de_normalise():
    """Test de-normalisation function."""
    train_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_y = np.array([[10.0], [20.0]])

    lower_bound = {"x1": 1.0, "x2": 2.0}
    upper_bound = {"x1": 3.0, "x2": 4.0}

    train_data = Dataset(
        dataset_type="train",
        x_data=train_x,
        y_data=train_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )

    train_data.add_normalisation_params(
        lower_bound=lower_bound, upper_bound=upper_bound
    )
    train_data.normalise()
    normalisation_stats = train_data.get_stats()

    prediction_mean = np.array([[0.0], [1.0]])
    prediction_error = np.array([[1.0], [4.0]])

    predictions = Dataset(
        dataset_type="predictions",
        x_data=train_x,
        y_data=prediction_mean,
        y_errors=prediction_error,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )
    predictions.normalised = True

    predictions.set_stats(normalisation_stats)
    predictions.de_normalise()

    predictions_mean_de_normalised = predictions.y_data
    predictions_error_de_normalised = predictions.y_errors

    expected_mean = np.array([[15.0], [20.0]])
    expected_error = np.array([[5.0], [20.0]])

    assert np.allclose(predictions_mean_de_normalised, expected_mean)
    assert np.allclose(predictions_error_de_normalised, expected_error)


def test_double_normalisation():
    """Test double normalisation and transformation."""
    train_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_y = np.array([[10.0], [20.0]])

    lower_bound = {"x1": 1.0, "x2": 2.0}
    upper_bound = {"x1": 3.0, "x2": 4.0}

    train_data = Dataset(
        dataset_type="train",
        x_data=train_x,
        y_data=train_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )

    train_data.add_normalisation_params(
        lower_bound=lower_bound, upper_bound=upper_bound
    )
    train_data.normalise()
    train_data.normalise()

    train_x_normalised = train_data.x_data
    train_y_normalised = train_data.y_data

    assert np.allclose(train_x_normalised, [[0, 0], [1, 1]])
    assert np.allclose(train_y_normalised, [[-1], [1]])

    normalisation_stats = train_data.get_stats()

    test_x = np.array([[1.0, 3.0]])
    test_y = np.array([[30]])

    test_data = Dataset(
        dataset_type="test",
        x_data=test_x,
        y_data=test_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )
    test_data.set_stats(normalisation_stats)
    test_data.normalise()
    test_data.normalise()

    test_x_normalised = test_data.x_data
    test_y_normalised = test_data.y_data

    assert np.allclose(test_x_normalised, [[0, 0.5]])
    assert np.allclose(test_y_normalised, [[3]])


def test_double_de_normalisation():
    """Test double de-normalisation."""
    train_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_y = np.array([[10.0], [20.0]])

    lower_bound = {"x1": 1.0, "x2": 2.0}
    upper_bound = {"x1": 3.0, "x2": 4.0}

    train_data = Dataset(
        dataset_type="train",
        x_data=train_x,
        y_data=train_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )

    train_data.add_normalisation_params(
        lower_bound=lower_bound, upper_bound=upper_bound
    )
    train_data.normalise()
    normalisation_stats = train_data.get_stats()

    prediction_mean = np.array([[0.0], [1.0]])
    prediction_error = np.array([[1.0], [4.0]])

    predictions = Dataset(
        dataset_type="predictions",
        x_data=train_x,
        x_labels=["x1", "x2"],
        y_data=prediction_mean,
        y_errors=prediction_error,
        y_labels=["y"],
    )
    predictions.normalised = True

    predictions.set_stats(normalisation_stats)
    predictions.de_normalise()
    predictions.de_normalise()

    predictions_mean_de_normalised = predictions.y_data
    predictions_error_de_normalised = predictions.y_errors

    expected_mean = np.array([[15.0], [20.0]])
    expected_error = np.array([[5.0], [20.0]])

    assert np.allclose(predictions_mean_de_normalised, expected_mean)
    assert np.allclose(predictions_error_de_normalised, expected_error)


def test_normalisation_double_de_normalisation():
    """Test de-normalisation, normalisation and then de-normalisation of predictions."""
    train_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_y = np.array([[10.0], [20.0]])

    lower_bound = {"x1": 1.0, "x2": 2.0}
    upper_bound = {"x1": 3.0, "x2": 4.0}

    train_data = Dataset(
        dataset_type="train",
        x_data=train_x,
        y_data=train_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )

    train_data.add_normalisation_params(
        lower_bound=lower_bound, upper_bound=upper_bound
    )
    train_data.normalise()
    normalisation_stats = train_data.get_stats()

    prediction_mean = np.array([[0.0], [1.0]])
    prediction_error = np.array([[1.0], [4.0]])

    predictions = Dataset(
        dataset_type="predictions",
        x_data=train_x,
        y_data=prediction_mean,
        y_errors=prediction_error,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )
    predictions.normalised = True

    predictions.set_stats(normalisation_stats)

    predictions.de_normalise()
    predictions.de_normalise()

    predictions_mean_de_normalised = predictions.y_data
    predictions_error_de_normalised = predictions.y_errors

    expected_mean = np.array([[15.0], [20.0]])
    expected_error = np.array([[5.0], [20.0]])

    assert np.allclose(predictions_mean_de_normalised, expected_mean)
    assert np.allclose(predictions_error_de_normalised, expected_error)


def test_copy():
    """Test copy function."""
    train_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    train_y = np.array([[10.0], [20.0]])
    train_data = Dataset(
        dataset_type="train",
        x_data=train_x,
        y_data=train_y,
        x_labels=["x1", "x2"],
        y_labels=["y"],
    )

    train_copy = train_data.copy()

    assert np.allclose(train_copy.x_data, train_data.x_data)
    assert np.allclose(train_copy.y_data, train_data.y_data)
    assert train_copy.x_labels == train_data.x_labels

    train_data.y_data = np.array([[10.0], [10.0]])

    assert not np.allclose(train_copy.y_data, train_data.y_data)
