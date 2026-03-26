from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PSOConfig:
    c1: float
    c2: float
    w: float
    particles: int
    dimensions: int
    bounds: list


@dataclass
class ExperimentConfigPFGS:
    method: str
    test_set_types: List[str]

    def __post_init__(self):
        if self.method not in {"PFGS"}:
            raise ValueError(f"Invalid method {self.method}")

        if self.test_set_types not in {"Design Space Only", "Entire Domain"}:
            raise ValueError(f"Invalid test set type: {self.test_set_types}")


@dataclass
class ExperimentConfigBO:
    method: str
    acquisition_function: str
    alphas: Optional[List[float]]
    test_set_types: List[str]

    def __post_init__(self):
        if self.acquisition_function not in {"UCB", "EI"}:
            raise ValueError(
                f"Unknown acquisition function: {self.acquisition_function}"
            )

        if self.acquisition_function == "UCB":
            if not self.alphas:
                raise ValueError(f"No range of alpha values provided")

        if self.acquisition_function == "EI":
            if self.alphas is not None:
                raise ValueError(f"Alpha values must not be provided for EI experiment")

        if self.test_set_types not in {"Design Space Only", "Entire Domain"}:
            raise ValueError(f"Invalid test set type: {self.test_set_types}")
