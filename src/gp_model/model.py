"""GP model: botorch's SingleTaskGP."""

import torch
import numpy as np
from numpy.typing import NDArray
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan

from src.data.data_models import Dataset


class GPModel:
    """
    Gaussian Process model with LBFGS optimizer.
    """

    def __init__(self):
        self.model = None
        self.mll = None
        torch.manual_seed(11)

    def train(self, dataset: Dataset):
        """
        Train GP on training dataset.

        Parameters:
            dataset (Dataset): The training dataset.
        """
        train_x = torch.tensor(dataset.x_data, dtype=torch.float64)
        train_y = torch.tensor(dataset.y_data, dtype=torch.float64)

        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(-1)

        self.model = SingleTaskGP(train_X=train_x, train_Y=train_y)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        self.model.likelihood.noise_covar.register_constraint(
            "raw_noise", GreaterThan(1e-5)
        )

        self.model.train()
        self.model.likelihood.train()
        fit_gpytorch_mll(self.mll)

    def predict(self, x: NDArray[np.float64]) -> Dataset:
        """
        Predict output based on test set.

        Parameters:
            x (NDArray[np.float64]): test set inputs.

        Returns:
            predictions (Dataset): mean and variance predictions.
        """
        test_x = torch.tensor(x, dtype=torch.float64)

        self.model.eval()
        self.model.likelihood.eval()

        with torch.no_grad():
            posterior = self.model.posterior(test_x, observation_noise=False)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            var = posterior.variance.squeeze(-1).cpu().numpy()

        predictions = Dataset(dataset_type="predictions", y_data=mean, y_errors=var)
        predictions.standardised = True

        return predictions
