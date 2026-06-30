"""Unit tests for investigation space"""

import pytest

from data.investigation_space import InvestigationSpace, InvestigationSpaceVariable


def test_investigation_space_properties():
    investigation_space = InvestigationSpace(
        variables=[
            InvestigationSpaceVariable(
                label="temperature",
                lower_bound=20,
                upper_bound=100,
            ),
            InvestigationSpaceVariable(
                label="pressure",
                lower_bound=1,
                upper_bound=10,
            ),
        ]
    )

    assert investigation_space.labels == ["temperature", "pressure"]

    assert investigation_space.lower_bounds == {
        "temperature": 20,
        "pressure": 1,
    }

    assert investigation_space.upper_bounds == {
        "temperature": 100,
        "pressure": 10,
    }

    assert investigation_space.lower_bound_list == [20, 1]
    assert investigation_space.upper_bound_list == [100, 10]


def test_investigation_space_validation():
    with pytest.raises(ValueError):
        InvestigationSpaceVariable(
            label="temperature",
            lower_bound=100,
            upper_bound=20,
        )

    with pytest.raises(ValueError):
        InvestigationSpace(
            variables=[
                InvestigationSpaceVariable("temperature", 20, 100),
                InvestigationSpaceVariable("temperature", 30, 90),
            ]
        )
