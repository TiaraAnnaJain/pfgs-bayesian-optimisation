"""Unit tests for configuration loading."""

from src.config.config_loading import load_config


def test_config_loading():
    """Test configuration loading from TOML file for PSO and Experiment configurations."""

    pso, experiments = load_config("configs/config.toml")

    assert pso.particles == 200
    assert len(experiments) > 0


def test_config_loading_experiments():
    """Test configuration loading from TOML file to make sure all experiments are loaded correctly."""
    _, experiments = load_config("configs/config.toml")

    names = [exp.name for exp in experiments]
    assert "ucb" in names
    assert "ei" in names
    assert "pfgs" in names


def test_config_loading_alpha():
    """Test configuration loading from TOML file to make sure it correctly unpacks alpha values."""
    _, experiments = load_config("configs/config.toml")

    ucb = next(exp for exp in experiments if exp.name == "ucb")
    assert ucb.acquisition_function == "UCB"
    assert ucb.alphas is not None
