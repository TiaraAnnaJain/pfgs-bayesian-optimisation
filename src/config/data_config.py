"""Data models for configuration setup."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PSOConfig:
    c1: float
    c2: float
    w: float
    particles: int
    dimensions: int


@dataclass(frozen=True)
class NSGAConfig:
    population_size: int
    number_of_generations: int


@dataclass(frozen=True)
class ExperimentConfigPFGS:
    method: str
    test_set_types: List[str]
    utopia_point: List[float]

    def __post_init__(self):
        if self.method not in {"PFGS"}:
            raise ValueError(f"Invalid method {self.method}")

        allowed_types = {"Design Space Only", "Entire Domain"}

        if not all(t in allowed_types for t in self.test_set_types):
            raise ValueError(f"Invalid test set type(s): {self.test_set_types}")


@dataclass(frozen=True)
class ExperimentConfigBO:
    method: str
    acquisition_function: str
    betas: Optional[List[float]]
    test_set_types: List[str]

    def __post_init__(self):
        print("DEBUG: New ExperimentConfigBO loaded")

        if self.acquisition_function not in {"UCB", "EI"}:
            raise ValueError(
                f"Unknown acquisition function: {self.acquisition_function}"
            )

        if self.acquisition_function == "UCB":
            if not self.betas:
                raise ValueError("No range of beta values provided")

        if self.acquisition_function == "EI":
            if self.betas is not None:
                raise ValueError("beta values must not be provided for EI experiment")

        allowed_types = {"Design Space Only", "Entire Domain"}

        if not all(t in allowed_types for t in self.test_set_types):
            raise ValueError(f"Invalid test set type(s): {self.test_set_types}")
