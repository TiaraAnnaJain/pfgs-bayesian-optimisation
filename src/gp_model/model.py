import torch
import numpy as np
from botorch.models import SingleTaskGP
from src.data.data_processing import to_tensor
from src.data.data_models import Dataset
from gpytorch.mlls import ExactMarginalLogLikelihood

class GPModel:
    def __init__(self):
        self.model = None
        self.mll = None
        self.optimizer = None

    def train(self, dataset: Dataset, verbose: bool = False):
        dataset.standardize()
        train_x, train_y = to_tensor(dataset.X, dataset.y)
        self.model = SingleTaskGP(train_X=train_x, train_Y=train_y)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)

        self.model.train()
        training_iter = 150

        for i in range(training_iter):
            self.optimizer.zero_grad()
            output = self.model(train_y)
            loss = -self.mll(output, train_y)
            loss.backward()

            if verbose:
                print(f"Iteration {i + 1}/{training_iter} - Loss: {loss.item()}")

            self.optimizer.step()

    def predict(self, dataset: Dataset, x):
        dataset.standardize()
        x_transformed = dataset.transform(x)
        test_x = to_tensor(x_transformed)

        self.model.eval()

        with torch.no_grad():
            posterior = self.model.posterior(test_x)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            var = posterior.variance.squeeze(-1).cpu().numpy()

        predictions = np.column_stack((mean, var))
        return predictions






