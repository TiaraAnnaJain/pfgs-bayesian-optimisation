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
