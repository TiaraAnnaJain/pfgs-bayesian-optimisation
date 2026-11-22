"""Tests for config validation."""

import pytest

from pfgs_bo.config import apply_overrides, load_config, validate_config


def base_config(**overrides):
    config = {
        "test_name": "test",
        "method": "PFGS",
        "num_iter": 5,
        "problem": "himmelblau",
        "train_samples": 5,
        "test_samples": 10,
        "train_seed": 0,
        "test_seed": 1,
        "model_seed": 11,
        "optimiser_seed": 0,
        "mean_discard_threshold": 0,
        "std_dev_discard_threshold": 0,
        "acquisition_function": "UCB",
        "beta": 1.96,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize("method", ["PFGS", "BO", "BO-PFGS", "constrained-BO"])
def test_valid_config_for_every_method(method):
    validate_config(base_config(method=method))


def test_pfgs_guided_bo_does_not_need_num_iter():
    config = base_config(method="BO-PFGS")
    del config["num_iter"]

    validate_config(config)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="not supported"):
        validate_config(base_config(method="random-search"))


def test_constrained_bo_needs_thresholds():
    config = base_config(method="constrained-BO")
    del config["mean_discard_threshold"]

    with pytest.raises(ValueError, match="discard thresholds"):
        validate_config(config)


def test_ucb_needs_beta():
    config = base_config(method="BO")
    del config["beta"]

    with pytest.raises(ValueError, match="beta"):
        validate_config(config)


def test_num_iter_must_be_positive_int():
    with pytest.raises(TypeError):
        validate_config(base_config(num_iter=0))


def test_unknown_problem_is_rejected():
    with pytest.raises(ValueError, match="not supported"):
        validate_config(base_config(problem="rosenbrock"))


def test_load_config_applies_command_line_overrides(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'problem = "himmelblau"\ntrain_seed = 1\nbeta = 1.96\n'
        "[nsga_parameters]\npopulation_size = 100\n"
    )

    config = load_config(
        [
            "--file",
            str(path),
            "--problem",
            "branin",
            "--train-seed",
            "4",
            "--set",
            "beta=2.5",
            "--set",
            "acquisition_function=EI",
            "--set",
            "nsga_parameters.population_size=20",
        ]
    )

    assert config["problem"] == "branin"
    assert config["train_seed"] == 4
    assert config["beta"] == 2.5
    assert config["acquisition_function"] == "EI"
    assert config["nsga_parameters"] == {"population_size": 20}


def test_override_must_be_key_value():
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({}, ["beta"])


def test_pfgs_needs_an_optimiser_seed():
    config = base_config(method="PFGS")
    del config["optimiser_seed"]

    with pytest.raises(TypeError, match="optimiser_seed"):
        validate_config(config)
