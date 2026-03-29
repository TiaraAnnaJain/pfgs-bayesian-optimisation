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

        allowed_types = {"Design Space Only", "Entire Domain"}

        if not all(t in allowed_types for t in self.test_set_types):
            raise ValueError(f"Invalid test set type(s): {self.test_set_types}")


@dataclass
class ExperimentConfigBO:
    method: str
    acquisition_function: str
    alphas: Optional[List[float]]
    test_set_types: List[str]

    def __post_init__(self):
        print("DEBUG: New ExperimentConfigBO loaded")

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

        allowed_types = {"Design Space Only", "Entire Domain"}

        if not all(t in allowed_types for t in self.test_set_types):
            raise ValueError(f"Invalid test set type(s): {self.test_set_types}")
