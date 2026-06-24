"""Himmelblau function."""

import numpy as np
from numpy.typing import NDArray


def himmelblau(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculates the modified Himmelblau function.

    Parameters:
        x (NDArray[np.float64]): input data, must be in [-5, 5] space

    Returns:
        NDArray[np.float64]: computed Himmelblau function.
    """
    if x.shape[1] != 2:
        raise ValueError("Input array is not 2-dimensional.")

    x_1, x_2 = x[:, 0], x[:, 1]
    y = ((x_1**2 + x_2 - 11) ** 2) + ((x_1 + x_2**2 - 7) ** 2) + x_1 - (2 * x_2)
    return np.exp(8 - 0.01 * y)

def branin(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculates the modified Branin function.

    Parameters:
        x (NDArray[np.float64]): input data.

    Returns:
        NDArray[np.float64]: computed modified Branin function.
    """
    if x.shape[1] != 2:
        raise ValueError("Input array is not 2-dimensional.")

    x_1, x_2 = x[:, 0], x[:, 1]

    a = 1.0
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    term1 = a * (x_2 - b * x_1 ** 2 + c * x_1 - r) ** 2
    term2 = s * (1 - t) * np.cos(x_1)

    y = 400 - (term1 + term2 + s)
    return y