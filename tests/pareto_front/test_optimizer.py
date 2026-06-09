"""Tests for GP Problem and NSGA-2 optimizer."""

import numpy as np

from src.pareto_front.gp_problem import GPProblem
from src.pareto_front.optimizer import NSGAOptimiser
from src.data.data_models import Dataset


class DummyModel:
    def __init__(self):
        self.x_labels = ["x1", "x2"]

    def predict(self, dataset):
        x = dataset.x_data

        return Dataset(
            dataset_type="predictions",
            x_labels=dataset.x_labels,
            x_data=x,
            y_labels=["mean"],
            y_data=x[:, 0].reshape(-1, 1),
            y_errors=x[:, 1].reshape(-1, 1),
        )


def test_gp_problem_evaluate():
    xl = np.array([1, 2])
    xu = np.array([3, 4])

    model = DummyModel()
    problem = GPProblem(model=model, bounds=(xl, xu))

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = {}

    problem._evaluate(x, out)

    expected = -np.array([[1.0, 2.0], [3.0, 4.0]])

    assert "F" in out
    assert out["F"].shape == (2, 2)
    assert np.allclose(out["F"], expected)


def test_gp_optimiser_shapes():
    model = DummyModel()

    xl = np.array([1, 2])
    xu = np.array([3, 4])

    problem = GPProblem(model=model, bounds=(xl, xu))

    optimiser = NSGAOptimiser(problem, pop_size=50, n_gen=2)
    scaled, unscaled, X = optimiser.run()

    assert scaled.shape[1] == 2
    assert unscaled.shape[1] == 2
    assert X.shape[1] == 2


def test_gp_optimiser_scaling_range():
    model = DummyModel()

    xl = np.array([1, 2])
    xu = np.array([3, 4])

    problem = GPProblem(model=model, bounds=(xl, xu))

    optimiser = NSGAOptimiser(problem, pop_size=50, n_gen=2)
    scaled, _, _ = optimiser.run()

    assert np.all(scaled >= 0)
    assert np.all(scaled <= 1)
