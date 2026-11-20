"""Tests for Particle Swarm Optimisation objective and optimiser."""

import numpy as np
from unittest.mock import MagicMock, patch

from pso.pso import PSOObjective, PSOptimiser


class DummyModel:
    x_labels = ["x1", "x2"]

    def predict(self, dataset):
        return dataset.x_data.sum(axis=1)


def dummy_acq(predictions):
    return predictions


def test_pso_objective_returns_negative_acquisition_values():
    objective = PSOObjective(
        model=DummyModel(),
        acq_fun=dummy_acq,
        device=None,
    )

    x = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    result = objective(x)

    assert np.allclose(result, np.array([-3.0, -7.0]))


def test_pso_objective_passes_acq_kwargs():
    def weighted_acq(predictions, beta):
        return beta * predictions

    objective = PSOObjective(
        model=DummyModel(),
        acq_fun=weighted_acq,
        device=None,
        beta=2.0,
    )

    x = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    result = objective(x)

    assert np.allclose(result, np.array([-6.0, -14.0]))


@patch("pso.pso.ps.single.GlobalBestPSO")
def test_pso_optimiser_runs_and_returns_cost_and_best_position(mock_global_best_pso):
    mock_optimizer = MagicMock()
    mock_optimizer.optimize.return_value = (-10.0, np.array([1.0, 2.0]))
    mock_global_best_pso.return_value = mock_optimizer

    objective = MagicMock()
    objective.stop_optimisation = False

    optimiser = PSOptimiser(
        objective=objective,
        n_particles=10,
        options={"c1": 0.5, "c2": 0.3, "w": 0.9},
        lower_bound=[0.0, 0.0],
        upper_bound=[1.0, 1.0],
    )

    cost, best_pos, stop_optimisation = optimiser.run()

    objective.assert_called_once_with(mock_optimizer.swarm.position)
    mock_optimizer.optimize.assert_called_once_with(objective, iters=100)
    assert cost == -10.0
    assert np.allclose(best_pos, np.array([1.0, 2.0]))
    assert stop_optimisation is False


@patch("pso.pso.ps.single.GlobalBestPSO")
def test_pso_optimiser_stops_when_no_point_survives_thresholds(mock_global_best_pso):
    mock_optimizer = MagicMock()
    mock_global_best_pso.return_value = mock_optimizer

    objective = MagicMock()
    objective.stop_optimisation = True

    optimiser = PSOptimiser(
        objective=objective,
        n_particles=10,
        options={"c1": 0.5, "c2": 0.3, "w": 0.9},
        lower_bound=[0.0, 0.0],
        upper_bound=[1.0, 1.0],
    )

    cost, best_pos, stop_optimisation = optimiser.run()

    mock_optimizer.optimize.assert_not_called()
    assert cost is None
    assert best_pos is None
    assert stop_optimisation is True
