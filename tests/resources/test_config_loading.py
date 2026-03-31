"""Unit tests for configuration loading."""

from src.config.config_loading import load_config


def test_config_loading():
    """Test configuration loading from TOML file for experiment and optimizer configurations."""

    experiment_ucb, pso_ucb = load_config("configs/bo_ucb_config.toml")
    experiment_ei, pso_ei = load_config("configs/bo_ei_config.toml")
    experiment_pfgs, nsga = load_config("configs/pfgs_config.toml")

    assert experiment_ucb.acquisition_function == "UCB"
    assert experiment_ucb.alphas is not None
    assert pso_ucb.particles == 200

    assert experiment_ei.acquisition_function == "EI"
    assert experiment_ei.alphas is None
    assert pso_ei.particles == 200

    assert experiment_pfgs.method == "PFGS"
    assert nsga.population_size == 200
