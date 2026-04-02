import torch
import numpy as np
from botorch.models import SingleTaskGP
from src.data.data_models import Dataset
from gpytorch.mlls import ExactMarginalLogLikelihood


class GPModel:
    def __init__(self):
        self.model = None
        self.mll = None
        self.optimizer = None

    def train(self, dataset: Dataset, verbose: bool = False):
        dataset.standardize()

        train_x = torch.tensor(dataset.x, dtype=torch.float64)
        train_y = torch.tensor(dataset.y, dtype=torch.float64)

        self.model = SingleTaskGP(train_X=train_x, train_Y=train_y)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)

        self.model.train()
        self.model.likelihood.train()

        training_iter = 150

        for i in range(training_iter):
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -self.mll(output, train_y)
            loss.backward()

            if verbose:
                print(f"Iteration {i + 1}/{training_iter} - Loss: {loss.item()}")

            self.optimizer.step()

    def predict(self, dataset: Dataset, x):
        x_transformed = dataset.transform(x, None)
        test_x = torch.tensor(x_transformed, dtype=torch.float64)

        self.model.eval()

        with torch.no_grad():
            posterior = self.model.posterior(test_x, observation_noise=True)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            var = posterior.variance.squeeze(-1).cpu().numpy()

        predictions = np.column_stack((mean, var))
        return predictions
