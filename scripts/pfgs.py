"""PFGS loop"""

import warnings
import numpy as np
from build.lib.botorch.exceptions.warnings import InputDataWarning
from src.data.data_processing import generate_data, himmelblau
from src.data.data_processing import Dataset
from src.gp_model.model import GPModel
from src.gp_model.analysis import compute_rrmse
from src.pareto_front.gp_problem import GPProblem
from src.pareto_front.optimizer import GPOptimiser
from src.pareto_front.optimal_point import OptimalPoints
from src.pareto_front.alpha_calculator import AlphaCalculator

warnings.filterwarnings(
    "ignore",
    message=r".*To copy construct from a tensor.*",
    category=UserWarning)

warnings.filterwarnings(
    "ignore",
    message="Data \\(input features\\) is not contained to the unit cube",
)

warnings.filterwarnings(
    "ignore",
    category=InputDataWarning,
)


def main():
    train_data, test_data, test_design_space = generate_data()
    model = GPModel()

    rrmse_plot = []
    alpha_upper = []
    alpha_lower = []

    num_iter = 95

    for i in range(num_iter):
        dataset = Dataset(train_data.x, train_data.y)

        train_set_mean, train_set_std = dataset.standardize()

        model.train(dataset)

        problem = GPProblem(
            model=model, train_set_mean=train_set_mean, train_set_std=train_set_std
        )

        optimizer = GPOptimiser(problem=problem, pop_size=100, n_gen=10)
        scaled_pareto_front, unscaled_pareto_front, input_values = optimizer.run()

        utopia_point = np.array([1.0, 1.0])
        optimal_point_evaluator = OptimalPoints(
            scaled_pareto_front=scaled_pareto_front,
            unscaled_pareto_front=unscaled_pareto_front,
            utopia_point=utopia_point,
        )
        closest_point, optimal_idx, optimal_point = (
            optimal_point_evaluator.closest_points(
                scaled_pareto_front=scaled_pareto_front
            )
        )

        alpha_calculator = AlphaCalculator(
            pareto_front=scaled_pareto_front, closest_point=closest_point
        )
        alpha_min, alpha_max = alpha_calculator.calculate_alpha()

        x_new = input_values[optimal_idx]
        x_new_raw = x_new * train_set_std + train_set_mean
        x_new_raw = x_new_raw.reshape(1, -1)

        y_new_raw = himmelblau(x_new_raw)

        y_new_raw = y_new_raw.reshape(1, 1)
        train_data.y = train_data.y.reshape(-1, 1)

        test_x, _ = dataset.transform(test_data.x)
        predictions_std = model.predict(test_x)

        predictions = dataset.de_standardize(predictions_std)
        rrmse = compute_rrmse(predictions, test_data.y)

        train_data.x = np.vstack((train_data.x, x_new_raw))
        train_data.y = np.vstack((train_data.y, y_new_raw))

        print(
            f"Iteration {i + 1}: alpha upper bound = {alpha_max}, alpha lower bound = {alpha_min}, RRMSE = {rrmse}, x_new: {x_new}"
        )

        rrmse_plot.append(rrmse)
        alpha_upper.append(alpha_max)
        alpha_lower.append(alpha_min)

if __name__ == "__main__":
    main()
