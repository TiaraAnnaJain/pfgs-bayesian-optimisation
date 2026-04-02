from src.gp_model.model import GPModel
from src.data.data_models import Dataset
import numpy as np

def make_dummy_dataset():
    x = np.linspace(0, 1, 20).reshape(-1, 1)
    y = np.sin(2 * np.pi * x).ravel()

    return Dataset(x=x, y=y)

def test_gp_model():
    dataset = make_dummy_dataset()
    model = GPModel()

    model.train(dataset)
    predictions = model.predict(dataset, dataset.x)

    assert predictions.shape[0] == dataset.x.shape[0]
    assert predictions.shape[1] == 2

    mean, var = predictions[:, 0], predictions[:, 1]

    assert mean.shape == (len(dataset.x),)
    assert var.shape == (len(dataset.x),)
    assert np.all(var >= 0)