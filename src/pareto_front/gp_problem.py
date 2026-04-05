"""Optimisation problem wrapper for GP model for pareto front generation."""

from typing import Any, Optional, Tuple
from pymoo.core.problem import Problem
import numpy as np
from numpy.typing import NDArray
import torch


class GPProblem(Problem):
    """
    Optimisation problem wrapper for a Gaussian Process model. It defines a 2 variable,
    2 objective problem where the GP model provides predictions.

    Parameters:
        model (Any): GP model that outputs a predicted mean and variance.
        train_set_mean (float): mean of training data.
        train_set_std (float): standard deviation of training data.
        bounds (Tuple[Tuple[float, float], Tuple[float, float]], optional): lower and
        upper bounds for the input variables. Defaults to the standardised space.
    """

    def __init__(
        self,
        model: Any,
        train_set_mean: float,
        train_set_std: float,
        bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> None:

        if bounds is None:
            lower_bound = (-5 - train_set_mean) / train_set_std
            upper_bound = (5 - train_set_mean) / train_set_std
            bounds = [lower_bound, upper_bound]
        xl, xu = bounds

        super().__init__(n_var=2, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

        self.model = model

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
        x_tensor = torch.from_numpy(x).float()
        predictions = self.model.predict(x_tensor)
        predictions = np.asarray(predictions, dtype=float).reshape(-1, 2)

        out["F"] = -predictions
