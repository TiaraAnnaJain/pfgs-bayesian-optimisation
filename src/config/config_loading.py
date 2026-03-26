from dataclasses import dataclass
import tomllib
from src.config.data_models import PSOConfig, ExperimentConfigBO, ExperimentConfigPFGS
from typing import Tuple


def load_config(
    path: str, experiment_name: str
) -> Tuple[PSOConfig, ExperimentConfigBO | ExperimentConfigPFGS]:
    """
    Loads configuration from config.toml file and converts to PSOConfig and ExperimentConfigBO or ExperimentConfigPFGS.

    Parameters:
    path (str): path to config.toml file
    experiment_name (str): name of experiment to load

    Returns:
    pso (PSOConfig): PSO configuration loaded from config.toml file
    experiments (list[ExperimentConfigBO | ExperimentConfigPFGS]): experiment configuration for either PFGS or BO loaded from config.toml file.
    """

    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    pso_raw = cfg["pso_parameters"]

    pso = PSOConfig(
        c1=pso_raw["c1"],
        c2=pso_raw["c2"],
        w=pso_raw["w"],
        particles=pso_raw["particles"],
        dimensions=pso_raw["dimensions"],
        bounds=pso_raw["bounds"],
    )

    experiments = cfg["experiments"]

    if experiment_name not in experiments:
        raise ValueError(f"Experiment {experiment_name} not found in config file")

    exp_raw = experiments[experiment_name]

    method = exp_raw["method"]

    if method == "BO":
        experiment = ExperimentConfigBO(
            method=exp_raw["method"],
            acquisition_function=exp_raw["acquisition_function"],
            alphas=exp_raw.get("alphas"),
            test_set_types=exp_raw["test_set_types"],
        )

    elif method == "PFGS":
        experiment = ExperimentConfigPFGS(
            method=exp_raw["method"],
            test_set_types=exp_raw["test_set_types"],
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return pso, experiment
