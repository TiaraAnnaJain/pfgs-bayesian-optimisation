from dataclasses import dataclass
import tomllib

@dataclass
class PSOConfig:
    c1: float
    c2: float
    w: float
    particles: int
    dimensions: int
    bounds: list


@dataclass
class ExperimentConfig:
    name: str
    method: str
    acquisition_function: str | None
    alphas: list | None
    test_set_types: list


def load_config(path: str) -> tuple[PSOConfig, list[ExperimentConfig]]:
    """
    Loads configuration from config.toml file and converts to PSOConfig and ExperimentConfig.

    Parameters:
    path (str): path to config.toml file

    Returns:
    pso (PSOConfig): PSO configuration loaded from config.toml file
    experiments (list[ExperimentConfig]): list of experiments loaded from config.toml file
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

    experiments = []
    for experiment in cfg["experiments"]:
        experiments.append(
            ExperimentConfig(
                name=experiment["name"],
                method=experiment["method"],
                acquisition_function=experiment.get("acquisition_function"),
                alphas=experiment.get("alphas"),
                test_set_types=experiment["test_set_types"],
            )
        )

    return pso, experiments
