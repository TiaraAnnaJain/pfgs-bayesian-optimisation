"""Tests for NSGA-2 optimizer."""

import numpy as np
import torch
from src.pareto_front.gp_problem import GPProblem
from src.pareto_front.optimizer import GPOptimiser


class DummyModel:
    def predict(self, x):
        return torch.stack([x[:, 0], x[:, 1]], dim=1)


def test_gp_optimiser_shapes():
    model = DummyModel()
    problem = GPProblem(model=model, train_set_mean=0, train_set_std=1)

    optimiser = GPOptimiser(problem, pop_size=50, n_gen=2)
    scaled, unscaled, X = optimiser.run()

    assert scaled.shape[1] == 2
    assert unscaled.shape[1] == 2
    assert X.shape[1] == 2


def test_gp_optimiser_scaling_range():
    model = DummyModel()
    problem = GPProblem(model=model, train_set_mean=0, train_set_std=1)

    optimiser = GPOptimiser(problem, pop_size=50, n_gen=2)
    scaled, _, _ = optimiser.run()

    assert np.all(scaled >= 0)
    assert np.all(scaled <= 1)
