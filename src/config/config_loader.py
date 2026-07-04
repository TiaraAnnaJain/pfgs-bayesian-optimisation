"""Loads config from toml files."""

from argparse import ArgumentParser
from pathlib import Path
import tomllib
from typing import Dict, Any

from data.simulation import SIMULATION_REGISTRY

PROJECT_ROOT = Path(__file__).parents[2]


def load_config() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads a TOML configuration file specified via command-line argument.

    Returns:
        config (dict): TOML configuration dictionary.
        files (dict): TOML configuration files.
    """
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Name of config file.")
    args = parser.parse_args()

    toml_path = PROJECT_ROOT / "configs" / args.file

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    config = {k: v for k, v in data.items() if k != "files"}
    files = data.get("files", {})

    return config, files


def resolve_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a flat, fully resolved config.

    Parameters:
         config (Dict[str, Any]): raw dict from tomllib.

    Returns:
        resolved (Dict[str, Any]): fully resolved config.
    """
    resolved = dict(config)

    if "simulation" in resolved:
        sim_name = resolved["simulation"]
        resolved["simulation"] = SIMULATION_REGISTRY[sim_name]

    return resolved


def validate_config(config: Dict[str, Any]) -> None:
    """Validations for config file."""
    if not config["test_name"]:
        raise ValueError("Please provide a test run name.")

    if config["method"] is None:
        raise ValueError("Please provide method from the following: BO, PFGS.")
    if config["method"] not in {"BO", "PFGS"}:
        raise ValueError(f"Method {config['method']} is not supported.")

    if config["num_iter"] is None:
        raise ValueError("Please provide number of iterations.")

    if not isinstance(config["num_iter"], int) or config["num_iter"] <= 0:
        raise TypeError("Number of iterations must be an integer greater than 0.")

    if config["method"] == "BO":
        if config["acquisition_function"] is None:
            raise ValueError(
                "Please provide acquisition function from the following: UCB, EI."
            )
        if config["acquisition_function"] not in ["UCB", "EI"]:
            raise ValueError("Acquisition function must be either UCB or EI.")
        if config["acquisition_function"] == "UCB":
            if config["beta"] is None:
                raise ValueError("Please provide beta parameter.")

    if config["simulation"] is None:
        raise ValueError(
            "Please provide simulation from the following: himmelblau, branin, ackley."
        )
    if config["simulation"] not in ("himmelblau", "branin", "ackley"):
        raise ValueError(f"Simulation {config['simulation']} is not supported.")

    if config["test_set_type"] is None:
        raise ValueError(
            "Please provide test set type from the following: Entire Space, Design Space."
        )
    if config["test_set_type"] not in ("Entire Space", "Design Space"):
        raise ValueError(f"Test set type {config['test_set_type']} is not supported.")

    if config["train_samples"] is None:
        raise ValueError("Please provide number of training samples.")
    if not isinstance(config["train_samples"], int) or config["train_samples"] <= 0:
        raise ValueError(
            "Number of training samples must be an integer greater than 0."
        )

    if config["test_samples"] is None:
        raise ValueError("Please provide number of test samples.")
    if not isinstance(config["test_samples"], int) or config["test_samples"] <= 0:
        raise ValueError("Number of test samples must be an integer greater than 0.")

    if config["test_set_type"] == "Design Space":
        if config["delta"] is None:
            raise ValueError("Please provide delta parameter.")
        if not isinstance(config["delta"], (float, int)) or config["delta"] <= 0:
            raise ValueError("Delta parameter must be a float greater than 0.")
        if config["maxima"] is None:
            raise ValueError("Please provide maxima of function.")

    if config["method"] == "PFGS":
        if (
            config["mean_discard_threshold"] is None
            or config["std_dev_discard_threshold"] is None
        ):
            raise ValueError(
                "Please provide mean and standard deviation discard thresholds."
            )

    for seed_name in ("train_seed", "test_seed"):
        if not isinstance(config[seed_name], int):
            raise TypeError(f"{seed_name} must be an integer.")
