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

    experiment_raw = cfg["experiment"]
    method = experiment_raw["method"]

    if method == "PFGS":
        if "nsga_parameters" not in cfg:
            raise ValueError("PFGS experiment requires [nsga_parameters]")

        experiment = ExperimentConfigPFGS(
            method=experiment_raw["method"],
            test_set_types=experiment_raw["test_set_types"],
            utopia_point=experiment_raw["utopia_point"],
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
            method=experiment_raw["method"],
            acquisition_function=experiment_raw["acquisition_function"],
            alphas=experiment_raw.get("alphas"),
            test_set_types=experiment_raw["test_set_types"],
        )

        pso_raw = cfg["pso_parameters"]
        optimizer = PSOConfig(
            c1=pso_raw["c1"],
            c2=pso_raw["c2"],
            w=pso_raw["w"],
            particles=pso_raw["particles"],
            dimensions=pso_raw["dimensions"],
            bounds=pso_raw["bounds"],
        )

    else:
        raise ValueError(f"Unknown method {method}")

    return experiment, optimizer
