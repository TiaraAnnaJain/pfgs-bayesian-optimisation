"""Adds and loads beta values to and from json file"""

from typing import List, Dict
import json


def to_json(beta_min: List[float], beta_max: List[float]) -> None:
    """Takes beta ranges and writes them into json file."""
    beta_range = {
        "beta_min": beta_min,
        "beta_max": beta_max,
    }

    with open("beta.json", "w") as file:
        json.dump(beta_range, file)


def from_json(json_file: str) -> Dict[str, List[float]]:
    """Takes json file with beta ranges and returns beta values."""
    with open(json_file, "r") as file:
        beta_range = json.load(file)

    return beta_range
