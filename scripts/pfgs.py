"""PFGS loop"""

import warnings

import numpy as np
from build.lib.botorch.exceptions.warnings import InputDataWarning
import matplotlib.pyplot as plt

from src.data.data_processing import generate_data
from src.data.simulation import himmelblau
from src.data.data_processing import Dataset
from src.gp_model.model import GPModel
from src.gp_model.analysis import compute_rrmse
from src.pareto_front.gp_problem import GPProblem
from src.pareto_front.optimizer import GPOptimiser
from src.pareto_front.optimal_point import OptimalPoints
from src.pareto_front.alpha_calculator import AlphaCalculator
from src.gp_model.plotting import contour_gp_plot

warnings.filterwarnings(
    "ignore", message=r".*To copy construct from a tensor.*", category=UserWarning
)

warnings.filterwarnings(
    "ignore",
    message="Data \\(input features\\) is not contained to the unit cube",
)

warnings.filterwarnings(
    "ignore",
    category=InputDataWarning,
)


def main():
    train, test, test_design_space = generate_data()
    model = GPModel()

    rrmse_plot = []
    alpha_upper = []
    alpha_lower = []

    num_iter = 95

    for i in range(num_iter):
        train_data = Dataset(train.x, train.y)

        train_data.standardize()

        model.train(train_data)

        problem = GPProblem(
            model=model,
            train_set_mean=train_data.x_standardiser_mean,
            train_set_std=train_data.x_standardiser_std,
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
        x_new_raw = (
            x_new * train_data.x_standardiser_std + train_data.x_standardiser_mean
        )
        x_new_raw = x_new_raw.reshape(1, -1)

        y_new_raw = himmelblau(x_new_raw)

        y_new_raw = y_new_raw.reshape(1, 1)
        train.y = train.y.reshape(-1, 1)

        test_x, _ = train_data.transform(test.x)
        predictions_std = model.predict(test_x)

        predictions = train_data.de_standardize(predictions_std)
        rrmse = compute_rrmse(predictions, test.y)

        train.x = np.vstack((train.x, x_new_raw))
        train.y = np.vstack((train.y, y_new_raw))

        final_dataset = train_data

        print(
            f"Iteration {i + 1}: alpha upper bound = {alpha_max}, alpha lower bound = {alpha_min}, RRMSE = {rrmse}, x_new: {x_new_raw}"
        )

        rrmse_plot.append(rrmse)
        alpha_upper.append(alpha_max)
        alpha_lower.append(alpha_min)

    feasible_alpha_max_nonzero = [alpha for alpha in alpha_upper if alpha != 0]
    feasible_alpha_min = max(alpha_lower)
    feasible_alpha_max = min(feasible_alpha_max_nonzero)
    feasible_alpha = [feasible_alpha_min, feasible_alpha_max]
    if feasible_alpha_max >= feasible_alpha_min:
        print(f"feasible range of alpha values: {feasible_alpha}")
    else:
        print("There is no feasible range of alpha values")

    x = range(len(alpha_upper))

    train_input_1, train_input_2, x1, x2, pred_mean_train = contour_gp_plot(
        train.x, model, final_dataset
    )

    fig_alpha = plt.figure()
    for x_i, alpha_upper_i, alpha_lower_i in zip(x, alpha_upper, alpha_lower):
        plt.plot([x_i, x_i], [alpha_lower_i, alpha_upper_i], color="blue")
    plt.xlabel("iterations")
    plt.ylabel("alpha")

    fig_rrmse = plt.figure()
    plt.plot(x, rrmse_plot)
    plt.title("RRMSE over iterations")
    plt.xlabel("iterations")
    plt.ylabel("RRMSE")

    gp_fig_train = plt.figure()
    cs = plt.contourf(
        x1.numpy(), x2.numpy(), pred_mean_train, levels=50, cmap="viridis_r"
    )

    plt.scatter(train_input_1, train_input_2, c="r")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar(cs)
    plt.title("GP Predictive Mean post iterations with training points")

    plt.show()


if __name__ == "__main__":
    main()
