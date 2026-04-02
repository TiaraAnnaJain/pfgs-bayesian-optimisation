import numpy as np
from sklearn.metrics import mean_squared_error


class Analysis:
    def __init__(self, predictions):
        self.predictions = predictions

    def compute_rmse(self, y_true):
        std_dev = np.std(y_true)
        rmse = np.sqrt(mean_squared_error(y_true, self.predictions[:, 0]))
        rmse_relative = (
            np.sqrt(mean_squared_error(y_true, self.predictions[:, 0])) / std_dev
        )
        return rmse, rmse_relative
