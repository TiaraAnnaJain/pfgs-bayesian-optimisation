"""Tests for the shared experiment building blocks."""

from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pytest

from pfgs_bo.data.investigation_space import (
    InvestigationSpace,
    InvestigationSpaceVariable,
)
from pfgs_bo.data.simulation import himmelblau
from pfgs_bo.experiment import (
    RunHistory,
    add_observation,
    build_datasets,
    evaluate_surrogate,
    fit_surrogate,
    log_iteration,
    unit_bounds,
)
from pfgs_bo.model import GPModel


@pytest.fixture
def space():
    return InvestigationSpace(
        variables=[
            InvestigationSpaceVariable(label="x1", lower_bound=-5.0, upper_bound=5.0),
            InvestigationSpaceVariable(label="x2", lower_bound=-5.0, upper_bound=5.0),
        ],
        outputs=["y"],
    )


@pytest.fixture
def config():
    return {
        "simulation": himmelblau,
        "train_samples": 6,
        "train_seed": 0,
        "test_samples": 20,
        "test_seed": 1,
        "delta": 1.0,
        "maxima": [[3.0, 2.0], [-2.805118, 3.131312]],
    }


def test_unit_bounds_match_number_of_variables(space):
    lower, upper = unit_bounds(space)

    assert lower == [0.0, 0.0]
    assert upper == [1.0, 1.0]


def test_build_datasets_shapes(config, space):
    train_data, test_sets = build_datasets(config, space)

    assert train_data.dataset_type == "train"
    assert train_data.x_data.shape == (6, 2)
    assert set(test_sets) == {"Entire Space", "Design Space"}
    assert test_sets["Entire Space"].x_data.shape == (20, 2)
    assert test_sets["Design Space"].x_data.shape == (20, 2)


def test_add_observation_appends_de_normalised_point(config, space):
    train_data, _ = build_datasets(config, space)
    model = GPModel()
    stats = fit_surrogate(model, train_data, space)
    assert train_data.normalised

    new_point = add_observation(train_data, np.array([0.5, 1.0]), stats, himmelblau)

    assert not train_data.normalised
    assert train_data.x_data.shape == (7, 2)
    assert train_data.y_data.shape == (7, 1)
    # 0.5 and 1.0 in the unit cube map to 0 and 5 in [-5, 5].
    np.testing.assert_allclose(new_point.x_data, [[0.0, 5.0]])
    np.testing.assert_allclose(train_data.x_data[-1], [0.0, 5.0])
    np.testing.assert_allclose(train_data.y_data[-1], himmelblau(new_point.x_data))


def test_evaluate_surrogate_returns_both_rrmse(config, space):
    train_data, test_sets = build_datasets(config, space)
    model = GPModel()
    stats = fit_surrogate(model, train_data, space)

    metrics = evaluate_surrogate(model, test_sets, stats)

    assert set(metrics) == {"entire_space_rrmse", "design_space_rrmse"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_run_history_columns():
    history = RunHistory()
    history.record(
        1.0, {"entire_space_rrmse": 0.5, "design_space_rrmse": 0.4}, beta=2.0
    )
    history.record(
        3.0, {"entire_space_rrmse": 0.3, "design_space_rrmse": 0.2}, beta=1.5
    )

    columns = history.to_dict()

    assert list(columns) == [
        "iteration",
        "entire_space_rrmse",
        "design_space_rrmse",
        "train_data_max",
        "beta",
    ]
    assert list(columns["iteration"]) == [0, 1]
    assert columns["train_data_max"] == [1.0, 3.0]
    assert columns["beta"] == [2.0, 1.5]


def test_log_iteration_logs_best_value_before_new_point(config, space):
    matplotlib.use("Agg")
    train_data, _ = build_datasets(config, space)
    best_observed = train_data.y_data.max()
    model = GPModel()
    stats = fit_surrogate(model, train_data, space)
    # A point at the Himmelblau maximum, so the new best is higher than before.
    new_point = add_observation(train_data, np.array([0.8, 0.7]), stats, himmelblau)
    tracker = MagicMock()

    log_iteration(
        tracker,
        1,
        model,
        train_data,
        new_point,
        {"entire_space_rrmse": 0.5, "design_space_rrmse": 0.4},
        best_observed,
    )

    assert tracker.log_iteration.call_args.kwargs["train_data_max"] == best_observed
