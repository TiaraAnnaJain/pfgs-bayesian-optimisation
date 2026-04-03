from src.gp_model.model import GPModel
from src.data.data_models import Dataset
import torch
import numpy as np


def make_dummy_dataset():
    x = np.linspace(0, 1, 20).reshape(-1, 1)
    y = np.sin(2 * np.pi * x).ravel()

    return Dataset(x=x, y=y)


def test_gp_model():
    dataset = make_dummy_dataset()
    dataset.standardize()

    model = GPModel()

    model.train(dataset)
    predictions = model.predict(dataset.x)

    mean, var = predictions[:, 0], predictions[:, 1]

    assert predictions.shape[0] == dataset.x.shape[0]
    assert predictions.shape[1] == 2
    assert mean.shape == (len(dataset.x),)
    assert var.shape == (len(dataset.x),)
    assert np.all(var >= 0)


def test_trainingdata_fit():
    dataset = make_dummy_dataset()
    dataset.standardize()

    model = GPModel()

    model.train(dataset)
    predictions = model.predict(dataset.x)

    mean = predictions[:, 0]
    _, y_true = dataset.transform_inverse(dataset.x, dataset.y)
    mse = ((mean - y_true) ** 2).mean()

    baseline = np.var(y_true)
    assert mse.item() < baseline


def test_gp_uncertainty_behavior():
    torch.manual_seed(0)

    dataset = make_dummy_dataset()
    dataset.standardize()

    model = GPModel()
    model.train(dataset)

    predictions_train = model.predict(dataset.x)
    var_train = predictions_train[:, 1].mean()

    test_x = np.array([[2.0]])
    predictions_test = model.predict(test_x)
    var_test = predictions_test[:, 1].mean()

    assert var_train < var_test
