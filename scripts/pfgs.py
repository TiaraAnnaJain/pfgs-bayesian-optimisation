"""PFGS loop"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.config_loader import load_config, resolve_config, validate_config
from data.data_models import Dataset
from data.data_processing import generate_data, generate_design_space_dataset
from data.investigation_space import InvestigationSpace
from gp_model.calculate_model_metrics import calculate_model_metrics
from gp_model.model import GPModel
from pareto_front.beta_calculator import calculate_beta
from pareto_front.gp_problem import GPProblem
from pareto_front.optimal_point import ParetoPointSelector
from pareto_front.optimizer import NSGAOptimiser
from plotting.gp_plotting import contour_slice_plot
from plotting.pareto_front_plot import plot_pareto_front
from tracking.csv_loader import to_csv
from tracking.json_loader import to_json
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

    entire_space = generate_data(
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
    design_space = generate_design_space_dataset(
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
            train_seed=resolved["train_seed"],
        )

        entire_space_rrmse_plot = []
        design_space_rrmse_plot = []
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

            datasets = {"Entire Space": entire_space, "Design Space": design_space}
            metrics = calculate_model_metrics(model, datasets, normalisation_stats)
            design_space_rrmse = metrics["Design Space RRMSE"]
            entire_space_rrmse = metrics["Entire Space RRMSE"]

            print(
                f"Iteration {i + 1}: beta upper bound = {beta_max}, beta lower bound = {beta_min}, "
                f"Entire space RRMSE = {entire_space_rrmse}, Design Space RRMSE: {design_space_rrmse}"
            )

            entire_space_rrmse_plot.append(entire_space_rrmse)
            design_space_rrmse_plot.append(design_space_rrmse)
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
                entire_space_rrmse=entire_space_rrmse,
                design_space_rrmse=design_space_rrmse,
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
        ax.plot(x, entire_space_rrmse_plot, label="RRMSE (Entire space)")
        ax.plot(x, design_space_rrmse_plot, label="RRMSE (Design space)")
        ax.legend()
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

        results = {
            "iteration": range(len(entire_space_rrmse_plot)),
            "entire_space_rrmse": entire_space_rrmse_plot,
            "design_space_rrmse": design_space_rrmse_plot,
            "train_data_max": train_data_max,
            "beta_upper": beta_upper,
            "beta_lower": beta_lower,
        }

        to_csv(results=results, folder="results/pfgs", seed=resolved["train_seed"])

        tracker.finish()


if __name__ == "__main__":
    main()
