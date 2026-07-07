"""PFGS loop"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.config_loader import load_config, resolve_config, validate_config
from config.json_loader import to_json
from data.data_models import Dataset
from data.data_processing import generate_data, generate_design_space_dataset
from data.investigation_space import InvestigationSpace
from gp_model.analysis import compute_rrmse
from gp_model.model import GPModel
from pareto_front.beta_calculator import calculate_beta
from pareto_front.gp_problem import GPProblem
from pareto_front.optimal_point import ParetoPointSelector
from pareto_front.optimizer import NSGAOptimiser
from plotting.gp_plotting import contour_slice_plot
from plotting.pareto_front_plot import plot_pareto_front
from tracking.mlflow_logging import MLflowTracker


def main():
    config, files = load_config()
    validate_config(config)

    BASE_DIR = Path(__file__).resolve().parents[1]

    json_path = BASE_DIR / files["investigation_space"]
    space = InvestigationSpace.from_json(json_path)

    resolved = resolve_config(config)
    simulation = resolved["simulation"]
    nsga_parameters = resolved["nsga_parameters"]

    lower_bound_normalised = [0, 0]
    upper_bound_normalised = [1.0, 1.0]

    x_labels = space.labels
    y_labels = space.outputs

    train_data = generate_data(
        simulation=simulation,
        n_samples=resolved["train_samples"],
        n_vars=len(x_labels),
        lower_bound=space.lower_bound_list,
        upper_bound=space.upper_bound_list,
        seed=resolved["train_seed"],
        dataset_type="train",
        x_labels=x_labels,
        y_labels=y_labels,
    )

    if resolved["test_set_type"] == "Entire Space":
        test_data = generate_data(
            simulation=simulation,
            n_samples=resolved["test_samples"],
            n_vars=len(x_labels),
            lower_bound=space.lower_bound_list,
            upper_bound=space.upper_bound_list,
            seed=resolved["test_seed"],
            dataset_type="test",
            x_labels=x_labels,
            y_labels=y_labels,
        )
    elif resolved["test_set_type"] == "Design Space":
        test_data = generate_design_space_dataset(
            simulation=simulation,
            n_samples=resolved["test_samples"],
            n_vars=len(x_labels),
            delta=resolved["delta"],
            seed=resolved["test_seed"],
            dataset_type="test",
            x_labels=x_labels,
            y_labels=y_labels,
            maxima=np.array(resolved["maxima"]),
        )

    utopia_point = np.array([1.0, 1.0])

    mean_discard_threshold = np.array(resolved["mean_discard_threshold"])
    std_dev_discard_threshold = np.array(resolved["std_dev_discard_threshold"])

    model = GPModel()

    with MLflowTracker(
        experiment_name=f"{simulation.__name__}-optimisation",
        run_name=f"{resolved['test_name']}",
    ) as tracker:
        tracker.log_resolved_config(resolved)

        tracker.log_experiment_setup(
            x_label=x_labels,
            y_label=y_labels,
            variables=space.variables,
            simulation=simulation.__name__,
        )

        rrmse_plot = []
        beta_upper = []
        beta_lower = []
        train_data_max = []

        num_iter = resolved["num_iter"]

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

            optimizer = NSGAOptimiser(
                problem=problem,
                pop_size=nsga_parameters["population_size"],
                n_gen=nsga_parameters["number_of_generations"],
            )
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

            beta_min, beta_max = calculate_beta(
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
            y_new = simulation(x_new.x_data).reshape(1, 1)

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
                f"Iteration {i + 1}: beta upper bound = {beta_max}, beta lower bound = {beta_min}, RRMSE = {rrmse}, x_new: {x_new.x_data}"
            )

            rrmse_plot.append(rrmse)
            beta_upper.append(beta_max)
            beta_lower.append(beta_min)
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
                iteration=i + 1,
                x_new=x_new,
            )

            tracker.log_iteration(
                step=i + 1,
                rrmse=rrmse,
                train_data_max=train_data.y_data.max(),
                beta_min=beta_min,
                beta_max=beta_max,
                pareto_fig=pareto_front_plot,
                gp_fig=gp_plot_iter,
            )

        feasible_beta_max_nonzero = [beta for beta in beta_upper if beta != 0]
        feasible_beta_min = max(beta_lower)
        feasible_beta_max = min(feasible_beta_max_nonzero)
        feasible_beta = [feasible_beta_min, feasible_beta_max]
        if feasible_beta_max >= feasible_beta_min:
            print(f"feasible range of beta values: {feasible_beta}")
        else:
            print("There is no feasible range of beta values")

        to_json(beta_min=beta_lower, beta_max=beta_upper)
        x = range(len(beta_upper))

        beta_fig, ax = plt.subplots()
        for x_i, beta_upper_i, beta_lower_i in zip(x, beta_upper, beta_lower):
            ax.plot([x_i, x_i], [beta_lower_i, beta_upper_i], color="blue")
        ax.set(title="Feasible beta ranges", xlabel="iterations", ylabel="beta")

        rrmse_fig, ax = plt.subplots()
        plt.plot(x, rrmse_plot)
        ax.set(title="RRMSE over iterations", xlabel="iterations", ylabel="RRMSE")

        train_data_max_fig, ax = plt.subplots()
        plt.plot(x, train_data_max)
        ax.set(
            title="Maximum value of train data",
            xlabel="iterations",
            ylabel="training data max",
        )

        gp_final_fig = contour_slice_plot(
            trained_model=model,
            train_dataset=train_data,
            x_plot_labels=x_labels,
            y_plot_label="y",
            iteration="Final",
        )

        tracker.log_final_results(
            final_gp_fig=gp_final_fig,
            rrmse_fig=rrmse_fig,
            beta_fig=beta_fig,
            train_data_max_fig=train_data_max_fig,
        )

        tracker.finish()


if __name__ == "__main__":
    main()
