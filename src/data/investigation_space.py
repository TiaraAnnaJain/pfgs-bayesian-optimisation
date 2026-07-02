from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class InvestigationSpaceVariable:
    label: str
    lower_bound: float
    upper_bound: float

    def __post_init__(self):
        if self.lower_bound >= self.upper_bound:
            raise ValueError(
                f"Lower bound must be smaller than upper bound for {self.label}"
            )


@dataclass(frozen=True)
class InvestigationSpace:
    variables: List[InvestigationSpaceVariable]

    def __post_init__(self):
        labels = [variable.label for variable in self.variables]

        if len(set(labels)) != len(labels):
            raise ValueError(f"Duplicate variable labels found.")

    @classmethod
    def from_json(cls, json_path: Path) -> "InvestigationSpace":
        data = json.loads(json_path.read_text())

        variables = [InvestigationSpaceVariable(**var) for var in data["variables"]]

        return cls(variables=variables)

    @property
    def labels(self) -> List[str]:
        return [variable.label for variable in self.variables]

    @property
    def lower_bounds(self) -> Dict[str, float]:
        return {variable.label: variable.lower_bound for variable in self.variables}

    @property
    def upper_bounds(self) -> Dict[str, float]:
        return {variable.label: variable.upper_bound for variable in self.variables}

    @property
    def lower_bound_list(self) -> List[float]:
        return list(self.lower_bounds.values())

    @property
    def upper_bound_list(self) -> List[float]:
        return list(self.upper_bounds.values())
