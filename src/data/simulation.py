"""Himmelblau function."""

import numpy as np
from numpy.typing import NDArray


def himmelblau(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculates the modified Himmelblau function."""
    if x.shape[1] != 2:
        raise ValueError("Input array is not 2-dimensional.")

    x_1, x_2 = x[:, 0], x[:, 1]
    y = ((x_1**2 + x_2 - 11) ** 2) + ((x_1 + x_2**2 - 7) ** 2) + x_1 - (2 * x_2)
    return np.exp(8 - 0.01 * y)
