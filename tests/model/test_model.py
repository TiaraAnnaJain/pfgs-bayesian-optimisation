"""Unit tests for GP Model."""

import torch
import numpy as np

from src.gp_model.model import GPModel
from src.data.data_models import Dataset

torch.manual_seed(0)


def make_dummy_dataset():
    x = np.linspace(0, 1, 20).reshape(-1, 1)
    y = np.sin(2 * np.pi * x)

    return Dataset(
        dataset_type="train", x_data=x, y_data=y, x_labels=["x"], y_labels=["y"]
    )


def test_gp_model():
    dataset = make_dummy_dataset()
    dataset.standardise()

    model = GPModel()
    model.train(dataset)

    predictions = model.predict(dataset)

    n = len(dataset.x_data)

    assert predictions.y_data.shape == (n,)
    assert predictions.y_errors.shape == (n,)
    assert np.all(predictions.y_errors >= 0)
    assert not np.isnan(predictions.y_data).any()

    assert predictions.dataset_type == "predictions"


def test_training_data_fit():
    dataset = make_dummy_dataset()
    dataset.standardise()

    model = GPModel()

    model.train(dataset)
    predictions = model.predict(dataset)

    dataset.de_standardise()
    mse = ((predictions.y_data.reshape(-1, 1) - dataset.y_data) ** 2).mean()

    baseline = np.var(dataset.y_data)
    assert mse.item() < baseline


def test_gp_uncertainty_behavior():
    dataset = make_dummy_dataset()
    dataset.standardise()

    model = GPModel()
    model.train(dataset)

    predictions_train = model.predict(dataset)
    var_train = predictions_train.y_errors.mean()

    test_x_data = np.array([[2.0]])
    test_x = Dataset(
        dataset_type="test",
        x_data=test_x_data,
        y_data=np.array([[2.0]]),
        x_labels=["x"],
    )
    predictions_test = model.predict(test_x)
    var_test = predictions_test.y_errors.mean()

    assert var_train < var_test
