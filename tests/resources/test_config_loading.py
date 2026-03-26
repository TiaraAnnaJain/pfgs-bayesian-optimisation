"""Unit tests for configuration loading."""

from src.config.config_loading import load_config


def test_config_loading():
    """Test configuration loading from TOML file for PSO and Experiment configurations."""

    pso, experiments = load_config("configs/config.toml")

    assert pso.particles == 200
    assert len(experiments) > 0

    names = [exp.name for exp in experiments]
    assert "ucb" in names
    assert "ei" in names
    assert "pfgs" in names

    ucb = next(exp for exp in experiments if exp.name == "ucb")
    assert ucb.acquisition_function == "UCB"
    assert ucb.alphas is not None
