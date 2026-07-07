"""Single step bayesian optimisation acquisition and candidate selection."""

import inspect
from typing import Any, List, Tuple, Union, Type

from data.data_models import Dataset

from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
import torch


class BoTorchOptimizer:
    """
    Botorch acquisition function and its optimiser to suggest the next point(s) to
    evaluate given a fitted GP model and the current training data.

    Parameters:
        acq_fn (Type[AcquisitionFunction]): acquisition function to use.
        lower_bounds (List[float]): lower bounds of the search space.
        upper_bounds (List[float]): upper bounds of the search space.
        num_restarts (int): number of restarts for multi-start optimisation of the
            acquisition function.
        raw_samples (int): number of samples used to generate initial conditions for
            optimisation restarts.
        q (int): number of candidate points to suggest per call.
        **acquisition_function_kwargs: keyword arguments passed to acquisition function.
    """

    def __init__(
        self,
        acq_fn: Type[AcquisitionFunction],
        lower_bounds: List[float],
        upper_bounds: List[float],
        num_restarts: int = 10,
        raw_samples: int = 512,
        q: int = 1,
        **acq_kwargs,
    ):
        self.bounds = torch.tensor([lower_bounds, upper_bounds])

        self.acq_fn = acq_fn
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.q = q
        self.acq_kwargs = acq_kwargs

    def _build_acq_fn(
        self, model: Any, train_data: Dataset
    ) -> Union[AcquisitionFunction, None]:
        """
        Instantiate the configured acquisition function for the given model.

        Parameters:
            model (Any): fitted GP model.
            train_data (Dataset): training data.

        Return:
            acqf (AcquisitionFunction): acquisition function.
        """
        auto_kwargs = {
            "model": model.model,
            "x_baseline": train_data.x_data,
            "best_f": train_data.y_data.max(),
        }

        sig = inspect.signature(self.acq_fn.__init__)
        accepted = set(sig.parameters.keys())

        kwargs = {k: v for k, v in auto_kwargs.items() if k in accepted}
        kwargs.update(self.acq_kwargs)

        return self.acq_fn(**kwargs)

    def optimize(
        self, model: Any, train_data: Dataset
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Suggest the next candidate point to evaluate.

        Parameters:
            model (Any): fitted GP model.
            train_data (Dataset): training data.

        Returns:
            candidate (torch.Tensor): Selected next point to evaluate.
            acq_val (torch.Tensor): Acquisition function value at the selected
                candidate.
        """
        acqf = self._build_acq_fn(model, train_data)
        candidates, acqf_val = optimize_acqf(
            acqf,
            bounds=self.bounds,
            q=self.q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )

        return candidates, acqf_val
