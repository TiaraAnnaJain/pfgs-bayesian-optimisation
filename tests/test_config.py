"""Tests for config validation."""

import pytest

from pfgs_bo.config import validate_config


def base_config(**overrides):
    config = {
        "test_name": "test",
        "method": "PFGS",
        "num_iter": 5,
        "simulation": "himmelblau",
        "train_samples": 5,
        "test_samples": 10,
        "train_seed": 0,
        "test_seed": 1,
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
