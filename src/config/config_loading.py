"""Loading function for TOML file."""

import tomllib

from typing import Tuple, Union
from src.config.data_config import (
    PSOConfig,
    ExperimentConfigBO,
    ExperimentConfigPFGS,
    NSGAConfig,
)


def load_config(
    path: str,
) -> Tuple[
    Union[ExperimentConfigBO, ExperimentConfigPFGS], Union[NSGAConfig, PSOConfig]
]:
    """Loads experiment configuration from TOML file.

    Parameters:
        path (str): Path to TOML file.

    Returns:
        experiment (Union[ExperimentConfigBO, ExperimentConfigPFGS]): Experiment configuration.
        optimizer (Union[NSGAConfig, PSOConfig]): Optimizer configuration.
    """

    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    method = cfg["method"]
    sweep_raw = cfg["sweep"]

    if method == "PFGS":
        if "nsga_parameters" not in cfg:
            raise ValueError("PFGS experiment requires [nsga_parameters]")

        experiment = ExperimentConfigPFGS(
            method=cfg["method"],
            test_set_types=sweep_raw["test_set_types"],
            utopia_point=cfg["utopia_point"],
        )

        nsga_raw = cfg["nsga_parameters"]
        optimizer = NSGAConfig(
            population_size=nsga_raw["population_size"],
            number_of_generations=nsga_raw["number_of_generations"],
        )

    elif method == "BO":
        if "pso_parameters" not in cfg:
            raise ValueError("BO experiment requires [pso_parameters]")

        experiment = ExperimentConfigBO(
            method=cfg["method"],
            acquisition_function=cfg["acquisition_function"],
            betas=sweep_raw.get("betas"),
            test_set_types=sweep_raw["test_set_types"],
        )

        pso_raw = cfg["pso_parameters"]
        optimizer = PSOConfig(
            c1=pso_raw["c1"],
            c2=pso_raw["c2"],
            w=pso_raw["w"],
            particles=pso_raw["particles"],
            dimensions=pso_raw["dimensions"],
        )

    else:
        raise ValueError(f"Unknown method {method}")

    return experiment, optimizer
