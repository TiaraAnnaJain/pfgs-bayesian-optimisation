"""BO loop with beta from PFGS"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config.config_loader import load_config, resolve_config
from data.investigation_space import InvestigationSpace
from data.data_processing import generate_data
from data.data_models import Dataset
from gp_model.model import GPModel
from gp_model.analysis import compute_rrmse
from bayesian_optimisation.acquisition_functions import compute_ucb
from bayesian_optimisation.pso import PSOObjective, PSOptimiser
from plotting.gp_plotting import contour_slice_plot
from tracking.mlflow_logging import MLflowTracker
from config.json_loader import from_json
from bayesian_optimisation.beta_pfgs import beta_ranges


def main():
    config, files = load_config()

    BASE_DIR = Path(__file__).resolve().parents[1]

    json_path = BASE_DIR / files["investigation_space"]
    space = InvestigationSpace.from_json(json_path)

    resolved = resolve_config(config)
    simulation = resolved["simulation"]

    lower_bound_normalised = [0, 0]
    upper_bound_normalised = [1.0, 1.0]

    x_labels = space.labels
    y_labels = ["y"]

    betas = from_json("scripts/beta.json")
    beta_schedule = beta_ranges(betas)

    for name, beta_values in beta_schedule.items():

        train_data = generate_data(
            simulation=simulation,
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
            simulation=simulation,
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

        with MLflowTracker(
            experiment_name=f"bo-pfgs-{simulation.__name__}", run_name=f"{name}-2"
        ) as tracker:
            tracker.log_resolved_config(resolved)

            tracker.log_experiment_setup(
                x_label=x_labels,
                y_label=y_labels,
                variables=space.variables,
                simulation=simulation.__name__,
                beta_schedule=name,
            )

            rrmse_plot = []
            train_data_max = []

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

                objective = PSOObjective(model=model, acq_fun=compute_ucb, beta=beta)
                optimiser = PSOptimiser(
                    objective=objective,
                    lower_bound=lower_bound_normalised,
                    upper_bound=upper_bound_normalised,
                )

                cost, x_new_point = optimiser.run()

                train_data.de_normalise()

                x_new = Dataset(
                    dataset_type="recipe",
                    x_labels=x_labels,
                    x_data=x_new_point.reshape(1, -1),
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

                rrmse = compute_rrmse(
                    predictions.y_data.reshape(-1, 1), test_data.y_data
                )

                print(f"Iteration {i + 1}: RRMSE = {rrmse}, beta: {beta}")

                rrmse_plot.append(rrmse)
                train_data_max.append(train_data.y_data.max())

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
                    rrmse=rrmse,
                    train_data_max=train_data.y_data.max(),
                    gp_fig=gp_plot_iter,
                    curr_beta=beta,
                )

            x = range(len(rrmse_plot))

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

            tracker.finish()


if __name__ == "__main__":
    main()