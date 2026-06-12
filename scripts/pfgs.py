"""PFGS loop"""

import warnings

import numpy as np
from build.lib.botorch.exceptions.warnings import InputDataWarning
import matplotlib.pyplot as plt

from data.investigation_space import InvestigationSpace, InvestigationSpaceVariable
from src.data.data_processing import generate_data
from src.data.simulation import himmelblau
from src.data.data_processing import Dataset
from src.gp_model.model import GPModel
from src.gp_model.analysis import compute_rrmse
from src.pareto_front.gp_problem import GPProblem
from src.pareto_front.optimizer import NSGAOptimiser
from src.pareto_front.optimal_point import ParetoPointSelector
from src.pareto_front.alpha_calculator import calculate_alpha
from plotting.gp_plotting import contour_slice_plot
from plotting.pareto_front_plot import plot_pareto_front

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
    variables = [
        InvestigationSpaceVariable(label="x1", lower_bound=-5, upper_bound=5),
        InvestigationSpaceVariable(label="x2", lower_bound=-5, upper_bound=5),
    ]

    space = InvestigationSpace(variables=variables)

    lower_bound_normalised = [0, 0]
    upper_bound_normalised = [1.0, 1.0]

    x_labels = space.labels
    y_labels = ["y"]

    train_data = generate_data(
        n_samples=5,
        n_vars=2,
        lower_bound=space.lower_bound_list,
        upper_bound=space.upper_bound_list,
        seed=42,
        dataset_type="train",
        x_labels=x_labels,
        y_labels=y_labels,
    )
    test_data = generate_data(
        n_samples=100,
        n_vars=2,
        lower_bound=space.lower_bound_list,
        upper_bound=space.upper_bound_list,
        seed=11,
        dataset_type="test",
        x_labels=x_labels,
        y_labels=y_labels,
    )

    model = GPModel()

    rrmse_plot = []
    alpha_upper = []
    alpha_lower = []

    num_iter = 95

    for i in range(num_iter):
        train_data.add_normalisation_params(
            lower_bound=space.lower_bounds, upper_bound=space.upper_bounds
        )
        train_data.normalise()
        normalisation_stats = train_data.get_stats()

        model.train(train_data)

        problem = GPProblem(
            model=model,
            lower_bound=lower_bound_normalised,
            upper_bound=upper_bound_normalised,
        )

        optimizer = NSGAOptimiser(problem=problem, pop_size=100, n_gen=10)
        (
            scaled_pareto_front,
            unscaled_pareto_front,
            input_values,
            pareto_front_scaler,
        ) = optimizer.run()

        utopia_point = np.array([1.0, 1.0])
        optimal_point_evaluator = ParetoPointSelector(
            scaled_pareto_front=scaled_pareto_front,
            unscaled_pareto_front=unscaled_pareto_front,
            utopia_point=utopia_point,
        )
        (
            mask,
            filtered_scaled_pareto_front,
            filtered_unscaled_pareto_front,
            adjusted_thresholds,
        ) = optimal_point_evaluator.discard_areas(
            mean_discard_threshold=np.array([0.3]),
            variance_discard_threshold=np.array([0.3]),
        )
        closest_point, optimal_idx, optimal_point = (
            optimal_point_evaluator.closest_points(
                scaled_pareto_front=filtered_scaled_pareto_front,
                unscaled_pareto_front=filtered_unscaled_pareto_front,
            )
        )

        alpha_min, alpha_max = calculate_alpha(
            pareto_front=scaled_pareto_front, closest_point=closest_point
        )

        train_data.de_normalise()

        x_new = input_values[optimal_idx].reshape(1, -1)

        x_new = Dataset(
            dataset_type="recipe",
            x_labels=x_labels,
            x_data=x_new,
            normalised=True,
        )

        x_new.set_stats(normalisation_stats)

        x_new.de_normalise()
        y_new = himmelblau(x_new.x_data).reshape(1, 1)

        train_data.x_data = np.vstack([train_data.x_data, x_new.x_data])
        train_data.y_data = np.vstack([train_data.y_data, y_new])

        recipe = test_data.copy()
        recipe.dataset_type = "recipe"
        recipe.y_data = None
        recipe.y_labels = None

        recipe.set_stats(normalisation_stats)
        recipe.normalise()

        predictions = model.predict(recipe)

        predictions.set_stats(normalisation_stats)
        predictions.de_normalise()

        rrmse = compute_rrmse(predictions.y_data.reshape(-1, 1), test_data.y_data)

        print(
            f"Iteration {i + 1}: alpha upper bound = {alpha_max}, alpha lower bound = {alpha_min}, RRMSE = {rrmse}, x_new: {x_new.x_data}"
        )

        rrmse_plot.append(rrmse)
        alpha_upper.append(alpha_max)
        alpha_lower.append(alpha_min)

        plot_pareto_front(
            unscaled_pareto_front=unscaled_pareto_front,
            train_data=train_data,
            discard_thresholds=adjusted_thresholds,
            optimal_point=optimal_point,
            scaler=pareto_front_scaler,
        )

    feasible_alpha_max_nonzero = [alpha for alpha in alpha_upper if alpha != 0]
    feasible_alpha_min = max(alpha_lower)
    feasible_alpha_max = min(feasible_alpha_max_nonzero)
    feasible_alpha = [feasible_alpha_min, feasible_alpha_max]
    if feasible_alpha_max >= feasible_alpha_min:
        print(f"feasible range of alpha values: {feasible_alpha}")
    else:
        print("There is no feasible range of alpha values")

    x = range(len(alpha_upper))

    train_input_1, train_input_2, x1, x2, pred_mean_train = contour_slice_plot(
        trained_model=model,
        train_dataset=train_data,
        x_plot_labels=x_labels,
        y_plot_label="y",
    )

    plt.figure()
    for x_i, alpha_upper_i, alpha_lower_i in zip(x, alpha_upper, alpha_lower):
        plt.plot([x_i, x_i], [alpha_lower_i, alpha_upper_i], color="blue")
    plt.xlabel("iterations")
    plt.ylabel("alpha")

    plt.figure()
    plt.plot(x, rrmse_plot)
    plt.title("RRMSE over iterations")
    plt.xlabel("iterations")
    plt.ylabel("RRMSE")

    plt.figure()
    cs = plt.contourf(x1, x2, pred_mean_train, levels=50, cmap="viridis_r")

    plt.scatter(train_input_1, train_input_2, c="r")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar(cs)
    plt.title("GP Predictive Mean post iterations with training points")

    plt.show()


if __name__ == "__main__":
    main()
