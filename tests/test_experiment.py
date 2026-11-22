"""Tests for the shared experiment building blocks."""

from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pytest

from pfgs_bo.benchmarks import BENCHMARKS
from pfgs_bo.data.investigation_space import (
    InvestigationSpace,
    InvestigationSpaceVariable,
)
from pfgs_bo.data.simulation import himmelblau
from pfgs_bo.experiment import (
    RunHistory,
    add_observation,
    beta_file,
    build_datasets,
    evaluate_surrogate,
    fit_surrogate,
    log_iteration,
    results_csv,
    run_dir,
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
        "problem": "himmelblau",
        "benchmark": BENCHMARKS["himmelblau"],
        "simulation": himmelblau,
        "train_samples": 6,
        "train_seed": 0,
        "test_samples": 20,
        "test_seed": 1,
        "delta": 1.0,
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


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        (
            {
                "method": "PFGS",
                "mean_discard_threshold": 0,
                "std_dev_discard_threshold": 0,
            },
            "himmelblau/pfgs/mean0_std0",
        ),
        (
            {"method": "BO", "acquisition_function": "UCB", "beta": 1.96},
            "himmelblau/bo/UCB_beta1.96",
        ),
        ({"method": "BO", "acquisition_function": "EI"}, "himmelblau/bo/EI"),
        (
            {
                "method": "constrained-BO",
                "acquisition_function": "EI",
                "mean_discard_threshold": 1200.0,
                "std_dev_discard_threshold": 0.5,
            },
            "himmelblau/constrained-bo/EI/mean1200_std0.5",
        ),
    ],
)
def test_run_dir_depends_on_method_settings(settings, expected, tmp_path):
    config = {"problem": "himmelblau", "results_dir": str(tmp_path), **settings}

    assert run_dir(config) == tmp_path / expected


def test_pfgs_and_pfgs_guided_bo_share_beta_file(tmp_path):
    shared = {
        "problem": "branin",
        "results_dir": str(tmp_path),
        "train_seed": 3,
        "mean_discard_threshold": 0,
        "std_dev_discard_threshold": 0,
    }
    pfgs = {**shared, "method": "PFGS"}
    pfgs_bo = {**shared, "method": "BO-PFGS"}

    expected = tmp_path / "branin/pfgs/mean0_std0/beta_train_seed_3.json"
    assert beta_file(pfgs) == beta_file(pfgs_bo) == expected
    assert results_csv(pfgs_bo, schedule="max") == (
        tmp_path / "branin/pfgs-bo/beta_from_mean0_std0/max/train_seed_3.csv"
    )


def test_pfgs_guided_bo_results_need_a_schedule(tmp_path):
    config = {
        "problem": "himmelblau",
        "method": "BO-PFGS",
        "mean_discard_threshold": 0,
        "std_dev_discard_threshold": 0,
    }

    with pytest.raises(ValueError, match="schedule"):
        run_dir(config)
