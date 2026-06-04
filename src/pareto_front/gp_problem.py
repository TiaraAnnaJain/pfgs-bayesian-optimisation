"""Optimisation problem wrapper for GP model for pareto front generation."""

from typing import Any, Tuple, List

from pymoo.core.problem import Problem
import numpy as np
from numpy.typing import NDArray

from src.data.data_models import Dataset

class GPProblem(Problem):
    """
    Optimisation problem wrapper for a Gaussian Process model. It defines a 2 variable,
    2 objective problem where the GP model provides predictions.

    Parameters:
        model (Any): GP model that outputs a predicted mean and variance.
        bounds (Tuple[NDArray[np.float64], NDArray[np.float64]]): lower and
            upper bounds for the input variables.
        x_labels (List[str]): labels for the input variables.
    """

    def __init__(
        self,
        model: Any,
        bounds: Tuple[NDArray[np.float64], NDArray[np.float64]],
        x_labels: List[str],
    ) -> None:

        xl, xu = bounds

        super().__init__(n_var=len(xl), n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

        self.model = model
        self.x_labels = x_labels

    def _evaluate(
        self, x: NDArray[np.float64], out: dict, *args: Any, **kwargs: Any
    ) -> None:
        """
         Evaluate the GP model at sampled points.

        Converts the optimizer-provided population `x` into a tensor, obtains
        predictions from the GP model, and stores the objective values in `out['F']`.

        Parameters:
            x (NDArray[np.float64]): Population points.
            out (dict): Dictionary of objective values.
            *args, **kwargs: Additional arguments required by pymoo.
        """
        candidate_dataset = Dataset(
            dataset_type="test",
            x_data=x,
            x_labels=self.x_labels,
            normalised=True,
        )

        predictions = self.model.predict(candidate_dataset)

        negated_predictions = -predictions
        predictions_mean, predictions_var = (
            negated_predictions.y_data,
            negated_predictions.y_errors,
        )

        out["F"] = np.column_stack([predictions_mean, predictions_var])
