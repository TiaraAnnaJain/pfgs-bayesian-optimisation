import numpy as np

from src.data.data_models import Dataset
from src.data.investigation_space import InvestigationSpace


def test_investigation_space_normalised_bounds():
    train_dataset = Dataset(
        dataset_type="train",
        x_labels=["x1", "x2"],
        x_data=np.array([[0.0, 10.0], [100.0, 20.0]]),
        y_labels=["y"],
        y_data=np.array([[10.0], [20.0]]),
    )

    train_dataset.normalise()

    investigation_space = InvestigationSpace(
        train_dataset=train_dataset,
        lower_bounds=np.array([25.0, 12.5]),
        upper_bounds=np.array([75.0, 17.5]),
    )

    lower, upper = investigation_space.normalised_bounds()

    expected_lower = np.array([0.25, 0.25])
    expected_upper = np.array([0.75, 0.75])

    assert np.allclose(lower, expected_lower)
    assert np.allclose(upper, expected_upper)
