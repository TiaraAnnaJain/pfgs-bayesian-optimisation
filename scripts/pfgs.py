"""PFGS loop"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from build.lib.botorch.exceptions.warnings import InputDataWarning

from data.investigation_space import InvestigationSpace, InvestigationSpaceVariable
from plotting.gp_plotting import contour_slice_plot
from plotting.pareto_front_plot import plot_pareto_front
from src.data.data_processing import Dataset, generate_data
from src.data.simulation import himmelblau, branin, ackley
from src.gp_model.analysis import compute_rrmse
from src.gp_model.model import GPModel
from src.pareto_front.alpha_calculator import calculate_alpha
from src.pareto_front.gp_problem import GPProblem
from src.pareto_front.optimal_point import ParetoPointSelector
from src.pareto_front.optimizer import NSGAOptimiser
from src.tracking.config_loader import load_config, resolve_config
from src.tracking.mlflow_logging import MLflowTracker

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
    config = load_config("pfgs_config.toml")

    resolved = resolve_config(config, {"test_set_type": "Entire Domain"})

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
        simulation=ackley,
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
        simulation=ackley,
        n_samples=100,
        n_vars=2,
        lower_bound=space.lower_bound_list,
        upper_bound=space.upper_bound_list,
        seed=11,
        dataset_type="test",
        x_labels=x_labels,
        y_labels=y_labels,
    )

    utopia_point = np.array([1.0, 1.0])

    mean_discard_threshold = np.array([14])
    std_dev_discard_threshold = np.array([0.3])

    model = GPModel()

    with MLflowTracker(experiment_name="ackley-pfgs", run_name="test-2") as tracker:
        tracker.log_resolved_config(resolved)

        tracker.log_experiment_setup(
            x_label=x_labels,
            y_label=y_labels,
            mean_discard_threshold=mean_discard_threshold,
            std_dev_discard_threshold=std_dev_discard_threshold,
            simulation_name="ackley",
            variables=variables,
        )

        rrmse_plot = []
        alpha_upper = []
        alpha_lower = []
        train_data_max = []

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
                pareto_front_points,
                input_values,
            ) = optimizer.run()

            pareto_front = train_data.copy()
            pareto_front.dataset_type = "pareto_front"
            pareto_front.y_data = pareto_front_points[:, 0]
            pareto_front.y_errors = pareto_front_points[:, 1]
            pareto_front.normalised = True
            pareto_front.de_normalise()

            optimal_point_evaluator = ParetoPointSelector(
                pareto_front=pareto_front,
                utopia_point=utopia_point,
            )
            (
                mask,
                filtered_pareto_front,
                discard_thresholds,
                stop_optimisation,
            ) = optimal_point_evaluator.discard_areas(
                mean_discard_threshold=mean_discard_threshold,
                std_dev_discard_threshold=std_dev_discard_threshold,
            )

            if stop_optimisation:
                break

            closest_point, optimal_idx, optimal_point, scaled_pareto_front = (
                optimal_point_evaluator.closest_points(
                    filtered_pareto_front=filtered_pareto_front
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
            y_new = ackley(x_new.x_data).reshape(1, 1)

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
            train_data_max.append(train_data.y_data.max())

            pareto_front_plot = plot_pareto_front(
                pareto_front=pareto_front,
                discard_thresholds=discard_thresholds,
                optimal_point=optimal_point,
                iteration=i + 1,
            )

            gp_plot_iter = contour_slice_plot(
                trained_model=model,
                train_dataset=train_data,
                x_plot_labels=x_labels,
                y_plot_label="y",
                iteration=i+1,
                x_new=x_new,
            )

            tracker.log_iteration(
                step=i+1,
                rrmse=rrmse,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                adjusted_thresholds=discard_thresholds,
                pareto_fig=pareto_front_plot,
                gp_fig=gp_plot_iter,
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

        alpha_fig, ax = plt.subplots()
        for x_i, alpha_upper_i, alpha_lower_i in zip(x, alpha_upper, alpha_lower):
            ax.plot([x_i, x_i], [alpha_lower_i, alpha_upper_i], color="blue")
        ax.set(title="Feasible alpha ranges", xlabel="iterations", ylabel="alpha")

        rrmse_fig, ax = plt.subplots()
        plt.plot(x, rrmse_plot)
        ax.set(title="RRMSE over iterations", xlabel="iterations", ylabel="RRMSE")

        train_data_max_fig, ax = plt.subplots()
        plt.plot(x, train_data_max)
        ax.set(title="Maximum value of train data", xlabel="iterations", ylabel="training data max")

        gp_final_fig = contour_slice_plot(
            trained_model=model,
            train_dataset=train_data,
            x_plot_labels=x_labels,
            y_plot_label="y",
            iteration="Final"
        )

        tracker.log_final_results(
            final_gp_fig=gp_final_fig,
            rrmse_fig=rrmse_fig,
            alpha_fig=alpha_fig,
            train_data_max_fig=train_data_max_fig,
        )

        tracker.finish()


if __name__ == "__main__":
    main()
