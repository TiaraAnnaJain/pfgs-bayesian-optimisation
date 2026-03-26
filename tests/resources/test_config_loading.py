"""Unit tests for configuration loading."""

from src.config.config_loading import load_config


def test_config_loading():
    """Test configuration loading from TOML file for PSO and Experiment configurations."""

    pso, experiment_ucb = load_config("configs/config.toml", "ucb")
    _, experiment_ei = load_config("configs/config.toml", "ei")
    _, experiment_pfgs = load_config("configs/config.toml", "pfgs")

    assert pso.particles == 200

    assert experiment_ucb.acquisition_function == "UCB"
    assert experiment_ucb.alphas is not None

    assert experiment_ei.acquisition_function == "EI"
    assert experiment_ei.alphas is None

    assert experiment_pfgs.method == "PFGS"
