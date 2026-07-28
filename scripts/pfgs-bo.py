"""BO loop with beta from PFGS"""

from pathlib import Path

from botorch.acquisition import qUpperConfidenceBound
import numpy as np
import matplotlib.pyplot as plt

from bayesian_optimisation.acq_fn_optimizer import BoTorchOptimizer
from config.config_loader import load_config, resolve_config
from data.investigation_space import InvestigationSpace
from data.data_processing import generate_data, generate_design_space_dataset
from data.data_models import Dataset
from gp_model.model import GPModel
from gp_model.calculate_model_metrics import calculate_model_metrics
from plotting.gp_plotting import contour_slice_plot
from tracking.csv_loader import to_csv
from tracking.mlflow_logging import MLflowTracker
from tracking.json_loader import from_json
from pfgs_guided_bo.beta_pfgs import beta_ranges


def main():
    config, files = load_config()

    BASE_DIR = Path(__file__).resolve().parents[1]

    json_path = BASE_DIR / files["investigation_space"]
    space = InvestigationSpace.from_json(json_path)

    resolved = resolve_config(config)
    simulation = resolved["simulation"]
    optimizer_parameters = resolved["optimizer_parameters"]

    lower_bound_normalised = [0, 0]
    upper_bound_normalised = [1.0, 1.0]

    x_labels = space.labels
    y_labels = space.outputs

    betas = from_json("beta.json")
    beta_schedule = beta_ranges(betas)

    for name, beta_values in beta_schedule.items():

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

        model = GPModel()

        with MLflowTracker(
            experiment_name=f"{simulation.__name__}-optimisation",
            run_name=f"{name}-{resolved['test_name']}",
        ) as tracker:
            tracker.log_resolved_config(resolved)

            tracker.log_experiment_setup(
                x_label=x_labels,
                y_label=y_labels,
                variables=space.variables,
                simulation=simulation.__name__,
                beta_schedule=name,
            )

            entire_space_rrmse_plot = []
            design_space_rrmse_plot = []
            train_data_max = []
            beta_csv = []

            default_beta = 1.96
            current_beta = default_beta

            for i, beta in enumerate(beta_values):

                if np.isfinite(beta):
                    current_beta = beta
                else:
                    beta = current_beta

                train_data.add_normalisation_params(
                    lower_bound=space.lower_bounds, upper_bound=space.upper_bounds
                )
                train_data.normalise()
                normalisation_stats = train_data.get_stats()

                model.train(train_data)

                optimizer = BoTorchOptimizer(
                    acq_fn=qUpperConfidenceBound,
                    lower_bounds=lower_bound_normalised,
                    upper_bounds=upper_bound_normalised,
                    num_restarts=optimizer_parameters["num_restarts"],
                    raw_samples=optimizer_parameters["raw_samples"],
                    q=optimizer_parameters["q"],
                    beta=beta,
                )
                x_new_point, _ = optimizer.optimize(model, train_data)

                train_data.de_normalise()

                x_new = Dataset(
                    dataset_type="recipe",
                    x_labels=x_labels,
                    x_data=x_new_point.reshape(1, -1).numpy(),
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
                    f"Iteration {i + 1}: Entire space RRMSE = {entire_space_rrmse}, "
                    f"Design Space RRMSE: {design_space_rrmse}"
                )

                entire_space_rrmse_plot.append(entire_space_rrmse)
                design_space_rrmse_plot.append(design_space_rrmse)
                train_data_max.append(train_data.y_data.max())
                beta_csv.append(beta)

                gp_plot_iter = contour_slice_plot(
                    trained_model=model,
                    train_dataset=train_data,
                    x_plot_labels=x_labels,
                    y_plot_label="y",
                    iteration=i + 1,
                    x_new=x_new,
                )

                tracker.log_iteration(
                    step=i,
                    entire_space_rrmse=entire_space_rrmse,
                    design_space_rrmse=design_space_rrmse,
                    train_data_max=train_data.y_data.max(),
                    gp_fig=gp_plot_iter,
                    curr_beta=beta,
                )

            x = range(len(entire_space_rrmse_plot))

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

            beta_fig, ax = plt.subplots()
            plt.plot(x, beta_values)
            ax.set(title="Value of beta", xlabel="iterations", ylabel="beta")

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
                "beta_value": beta_csv,
            }

            to_csv(
                results=results,
                folder=f"results/pfgs-bo/{name}",
                seed=resolved["train_seed"],
            )

            tracker.finish()


if __name__ == "__main__":
    main()
