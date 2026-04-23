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
    lb = [-5, -5]
    ub = [5, 5]

    train_data = generate_data(
        n_samples=5,
        n_vars=2,
        lower_bound=lb,
        upper_bound=ub,
        seed=42,
        dataset_type="train",
    )
    test_data = generate_data(
        n_samples=100,
        n_vars=2,
        lower_bound=lb,
        upper_bound=ub,
        seed=11,
        dataset_type="test",
    )
    model = GPModel()

    rrmse_plot = []
    alpha_upper = []
    alpha_lower = []

    num_iter = 95

    for i in range(num_iter):
        train_data.standardise()
        standardisation_stats = train_data.get_stats()

        model.train(train_data)
        train_data.de_standardise()

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

        x_new = input_values[optimal_idx].reshape(1, -1)
        x_new_real = (
            x_new * train_data.x_standardiser_std + train_data.x_standardiser_mean
        )
        y_new = himmelblau(x_new_real).reshape(1, 1)

        train_data.x_data = np.vstack([train_data.x_data, x_new_real])
        train_data.y_data = np.vstack([train_data.y_data, y_new])

        test_data.set_stats(standardisation_stats)
        test_data.standardise()

        predictions = model.predict(test_data.x_data)

        predictions.set_stats(standardisation_stats)
        predictions.de_standardise()

        test_data.de_standardise()

        rrmse = compute_rrmse(predictions.y_data, test_data.y_data)

        print(
            f"Iteration {i + 1}: alpha upper bound = {alpha_max}, alpha lower bound = {alpha_min}, RRMSE = {rrmse}, x_new: {x_new_real}"
        )

        rrmse_plot.append(rrmse)
        alpha_upper.append(alpha_max)
        alpha_lower.append(alpha_min)

    final_dataset = train_data

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
        train_data.x_data, model, final_dataset
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
