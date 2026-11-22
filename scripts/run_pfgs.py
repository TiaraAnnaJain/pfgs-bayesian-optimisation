"""Pareto Front Guided Sampling (PFGS)."""

import numpy as np

from pfgs_bo.experiment import (
    RunHistory,
    add_observation,
    beta_file,
    build_datasets,
    evaluate_surrogate,
    finish_run,
    fit_surrogate,
    load_experiment,
    log_iteration,
    optimiser_seeds,
    print_progress,
    results_csv,
    start_run,
    unit_bounds,
)
from pfgs_bo.model import GPModel
from pfgs_bo.pareto_front.beta import calculate_beta, feasible_beta_range
from pfgs_bo.pareto_front.generation import GPProblem, NSGAOptimiser
from pfgs_bo.pareto_front.selection import ParetoPointSelector
from pfgs_bo.plotting.pareto_front import plot_pareto_front
from pfgs_bo.plotting.results import plot_beta_ranges
from pfgs_bo.tracking.results_io import to_json


def main() -> None:
    """Runs PFGS with the config given on the command line."""
    config, space = load_experiment("PFGS")
    train_data, test_sets = build_datasets(config, space)
    lower_bound, upper_bound = unit_bounds(space)

    nsga_parameters = config["nsga_parameters"]
    utopia_point = np.array([1.0, 1.0])
    mean_discard_threshold = np.array(config["mean_discard_threshold"])
    std_dev_discard_threshold = np.array(config["std_dev_discard_threshold"])

    model = GPModel(seed=config["model_seed"])
    history = RunHistory()
    seeds = optimiser_seeds(config)

    with start_run(config, space) as tracker:
        for i in range(config["num_iter"]):
            step = i + 1
            best_observed = train_data.y_data.max()
            normalisation_stats = fit_surrogate(model, train_data, space)

            problem = GPProblem(
                model=model, lower_bound=lower_bound, upper_bound=upper_bound
            )
            optimizer = NSGAOptimiser(
                problem=problem,
                pop_size=nsga_parameters["population_size"],
                n_gen=nsga_parameters["number_of_generations"],
                seed=next(seeds),
            )
            pareto_front_points, input_values = optimizer.run()

            pareto_front = train_data.copy()
            pareto_front.dataset_type = "pareto_front"
            pareto_front.y_data = pareto_front_points[:, 0]
            pareto_front.y_errors = pareto_front_points[:, 1]
            pareto_front.normalised = True
            pareto_front.de_normalise()

            selector = ParetoPointSelector(
                pareto_front=pareto_front, utopia_point=utopia_point
            )
            _, filtered_pareto_front, discard_thresholds, stop_optimisation = (
                selector.discard_areas(
                    mean_discard_threshold=mean_discard_threshold,
                    std_dev_discard_threshold=std_dev_discard_threshold,
                )
            )
            if stop_optimisation:
                break

            closest_point, optimal_idx, optimal_point, scaled_pareto_front = (
                selector.closest_points(filtered_pareto_front=filtered_pareto_front)
            )

            beta_min, beta_max = calculate_beta(
                pareto_front=scaled_pareto_front, closest_point=closest_point
            )

            new_point = add_observation(
                train_data,
                input_values[optimal_idx],
                normalisation_stats,
                config["simulation"],
            )
            metrics = evaluate_surrogate(model, test_sets, normalisation_stats)
            history.record(
                best_observed, metrics, beta_upper=beta_max, beta_lower=beta_min
            )

            print_progress(step, metrics, beta_min=beta_min, beta_max=beta_max)
            log_iteration(
                tracker,
                step,
                model,
                train_data,
                new_point,
                metrics,
                best_observed,
                beta_min=beta_min,
                beta_max=beta_max,
                pareto_fig=plot_pareto_front(
                    pareto_front=pareto_front,
                    discard_thresholds=discard_thresholds,
                    optimal_point=optimal_point,
                    iteration=step,
                ),
            )

        beta_upper = history.extra.get("beta_upper", [])
        beta_lower = history.extra.get("beta_lower", [])

        feasible_range = feasible_beta_range(beta_lower, beta_upper)
        if feasible_range is not None:
            print(f"feasible range of beta values: {list(feasible_range)}")
        else:
            print("There is no feasible range of beta values")

        to_json(
            beta_min=beta_lower,
            beta_max=beta_upper,
            json_file=beta_file(config),
        )

        finish_run(
            tracker,
            model,
            train_data,
            history,
            csv_path=results_csv(config),
            beta_fig=plot_beta_ranges(beta_lower, beta_upper),
        )


if __name__ == "__main__":
    main()
