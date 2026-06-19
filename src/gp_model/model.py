"""GP model: botorch's SingleTaskGP."""

from typing import List, Optional

import torch
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
        self.x_labels: Optional[List[str]] = None
        self.y_labels: Optional[List[str]] = None
        torch.manual_seed(11)

    def train(self, train_dataset: Dataset):
        """
        Train GP on training dataset.

        Parameters:
            train_dataset (Dataset): The training dataset with dataset type "train".
        """
        if train_dataset.dataset_type != "train":
            raise ValueError(
                f"Dataset type must be 'train', got {train_dataset.dataset_type}"
            )

        self.x_labels = train_dataset.x_labels
        self.y_labels = train_dataset.y_labels

        train_x = torch.tensor(train_dataset.x_data, dtype=torch.float64)
        train_y = torch.tensor(train_dataset.y_data, dtype=torch.float64)

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

    def predict(self, dataset: Dataset) -> Dataset:
        """
        Predict output based on test set.

        Parameters:
            dataset (Dataset): copy of recipes for predictions, only contains x_data.

        Returns:
            predictions (Dataset): mean and standard deviation predictions.
        """

        if dataset.x_labels != self.x_labels:
            raise ValueError(
                f"Train and test datasets must have same x_labels {self.x_labels}."
            )
        dataset.check()

        test_x = torch.tensor(dataset.x_data, dtype=torch.float64)

        self.model.eval()
        self.model.likelihood.eval()

        with torch.no_grad():
            posterior = self.model.posterior(test_x, observation_noise=False)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            std_dev = posterior.variance.squeeze(-1).sqrt().cpu().numpy()

        dataset.dataset_type = "predictions"
        dataset.y_data = mean
        dataset.y_errors = std_dev
        dataset.y_labels = self.y_labels
        dataset.normalised = True
        dataset.check()

        return dataset
