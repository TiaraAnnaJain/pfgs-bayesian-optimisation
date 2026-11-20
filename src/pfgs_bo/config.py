"""Loads config from toml files."""

import tomllib
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from pfgs_bo.data.simulation import SIMULATION_REGISTRY

PROJECT_ROOT = Path(__file__).parents[2]


def load_config() -> tuple[dict[str, Any], dict[str, Any]]:
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


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Returns a flat, fully resolved config.

    Parameters:
         config (dict[str, Any]): raw dict from tomllib.

    Returns:
        resolved (dict[str, Any]): fully resolved config.
    """
    resolved = dict(config)

    if "simulation" in resolved:
        sim_name = resolved["simulation"]
        resolved["simulation"] = SIMULATION_REGISTRY[sim_name]

    return resolved


METHODS = ("PFGS", "BO", "BO-PFGS", "constrained-BO")
ACQUISITION_FUNCTIONS = ("UCB", "EI")


def _require(config: dict[str, Any], key: str, message: str) -> Any:
    """Returns ``config[key]``, or raises a ValueError with ``message`` if missing."""
    if config.get(key) is None:
        raise ValueError(message)
    return config[key]


def _require_positive_int(config: dict[str, Any], key: str, description: str) -> None:
    """Checks that ``config[key]`` is an integer greater than 0."""
    value = _require(config, key, f"Please provide {description}.")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(
            f"{description.capitalize()} must be an integer greater than 0."
        )


def validate_config(config: dict[str, Any]) -> None:
    """
    Checks that a raw config has every setting its method needs.

    Parameters:
        config (dict[str, Any]): raw config from ``load_config``.

    Raises:
        ValueError: if a setting is missing or has an unsupported value.
        TypeError: if a setting has the wrong type.
    """
    if not config.get("test_name"):
        raise ValueError("Please provide a test run name.")

    method = _require(
        config,
        "method",
        f"Please provide method from the following: {', '.join(METHODS)}.",
    )
    if method not in METHODS:
        raise ValueError(f"Method {method} is not supported.")

    if method != "BO-PFGS":
        _require_positive_int(config, "num_iter", "number of iterations")

    if method in ("BO", "constrained-BO"):
        acquisition_function = _require(
            config,
            "acquisition_function",
            "Please provide acquisition function from the following: UCB, EI.",
        )
        if acquisition_function not in ACQUISITION_FUNCTIONS:
            raise ValueError("Acquisition function must be either UCB or EI.")
        if acquisition_function == "UCB":
            _require(config, "beta", "Please provide beta parameter.")

    simulation = _require(
        config,
        "simulation",
        "Please provide simulation from the following: "
        f"{', '.join(SIMULATION_REGISTRY)}.",
    )
    if simulation not in SIMULATION_REGISTRY:
        raise ValueError(f"Simulation {simulation} is not supported.")

    _require_positive_int(config, "train_samples", "number of training samples")
    _require_positive_int(config, "test_samples", "number of test samples")

    if method in ("PFGS", "constrained-BO") and (
        config.get("mean_discard_threshold") is None
        or config.get("std_dev_discard_threshold") is None
    ):
        raise ValueError(
            "Please provide mean and standard deviation discard thresholds."
        )

    for seed_name in ("train_seed", "test_seed"):
        if not isinstance(config.get(seed_name), int):
            raise TypeError(f"{seed_name} must be an integer.")
