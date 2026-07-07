"""Contains filtering function for acquisition function for constrained BO"""

from typing import Dict, Any

import numpy as np
from numpy.typing import NDArray

from data.data_models import Dataset

def filter_predictions(preds: Dataset, mean_threshold: NDArray[np.float64], std_threshold: NDArray[np.float64], normalisation_stats: Dict[str, Any]) -> NDArray[np.float64]:
    """
    Creates a filter based on thresholds to apply to acquisition function.

    Parameters:
        preds (Dataset): GP predictions.
        mean_threshold (NDArray[np.float64]): mean threshold.
        std_threshold (NDArray[np.float64]): standard deviation threshold.
        normalisation_stats (Dict[str, Any]): normalisation stats.

    Returns:
        mask (NDArray[np.float64]): boolean mask of predictions.
    """
    preds_raw = preds.copy()
    preds_raw.set_stats(normalisation_stats)
    preds_raw.de_normalise()

    mean = preds_raw.y_data
    std = preds_raw.y_errors

    mask = (mean >= mean_threshold) & (std >= std_threshold)
    mask = np.asarray(mask).reshape(-1)

    return mask
