"""Tests for GP problem wrapper."""

import torch
import numpy as np
from src.pareto_front.gp_problem import GPProblem


class DummyModel:
    def predict(self, x):
        return torch.stack([x[:, 0], x[:, 1]], dim=1)


def test_gp_problem_bounds():
    mean = 2.0
    std = 2.0

    problem = GPProblem(model=None, train_set_mean=mean, train_set_std=std)

    expected_lower = (-5 - mean) / std
    expected_upper = (5 - mean) / std

    assert np.allclose(problem.xl, expected_lower)
    assert np.allclose(problem.xu, expected_upper)


def test_gp_problem_evaluate():
    model = DummyModel()
    problem = GPProblem(model=model, train_set_mean=0, train_set_std=1)

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = {}

    problem._evaluate(x, out)

    expected = -np.array([[1.0, 2.0], [3.0, 4.0]])

    assert "F" in out
    assert out["F"].shape == (2, 2)
    assert np.allclose(out["F"], expected)
