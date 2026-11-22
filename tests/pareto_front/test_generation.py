"""Tests for GP Problem and NSGA-2 optimizer."""

import numpy as np

from pfgs_bo.data.data_models import Dataset
from pfgs_bo.pareto_front.generation import GPProblem, NSGAOptimiser


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
    xl = [1, 2]
    xu = [3, 4]

    model = DummyModel()
    problem = GPProblem(model=model, lower_bound=xl, upper_bound=xu)

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = {}

    problem._evaluate(x, out)

    expected = -np.array([[1.0, 2.0], [3.0, 4.0]])

    assert "F" in out
    assert out["F"].shape == (2, 2)
    assert np.allclose(out["F"], expected)


def test_gp_optimiser_shapes():
    xl = [1, 2]
    xu = [3, 4]

    model = DummyModel()
    problem = GPProblem(model=model, lower_bound=xl, upper_bound=xu)

    optimiser = NSGAOptimiser(problem, pop_size=50, n_gen=2)
    unscaled, X = optimiser.run()

    assert unscaled.shape[1] == 2
    assert X.shape[1] == 2


class TradeOffModel(DummyModel):
    """Mean and standard deviation that conflict, so the front has many points."""

    def predict(self, dataset):
        x = dataset.x_data
        return Dataset(
            dataset_type="predictions",
            x_labels=dataset.x_labels,
            x_data=x,
            y_labels=["mean"],
            y_data=x[:, 0].reshape(-1, 1),
            y_errors=(1 - x[:, 0] + 0.1 * x[:, 1]).reshape(-1, 1),
        )


def test_nsga_optimiser_is_reproducible_with_a_seed():
    problem = GPProblem(model=TradeOffModel(), lower_bound=[0, 0], upper_bound=[1, 1])

    def run(seed):
        return NSGAOptimiser(problem, pop_size=20, n_gen=3, seed=seed).run()[1]

    np.testing.assert_array_equal(run(seed=7), run(seed=7))
    assert not np.array_equal(run(seed=7), run(seed=8))
