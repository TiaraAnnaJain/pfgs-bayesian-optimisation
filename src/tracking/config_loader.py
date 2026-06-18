"""Loads config from toml files."""

import tomllib
from typing import Dict, Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]


def load_config(file_name: str) -> Dict[str, Any]:
    toml_path = PROJECT_ROOT / "configs" / file_name
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


def resolve_config(config: Dict[str, Any], choices: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a flat, fully resolved config where every list key has been replaced by
    the single chosen value.

    Parameters:
         config (Dict[str, Any]): raw dict from tomllib.
         choices (Dict[str, Any]): chosen values.

    Returns:
        resolved (Dict[str, Any]): fully resolved config.
    """
    resolved = {k: v for k, v in config.items() if k != "sweep"}
    sweep = config.get("sweep", {})

    for k in sweep:
        if k not in choices:
            raise ValueError(f"No choice provided for sweep key: '{k}'")
        resolved[k] = choices[k]

    return resolved
