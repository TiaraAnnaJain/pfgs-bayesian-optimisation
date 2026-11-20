"""PFGS-guided Bayesian optimisation."""

import numpy as np
from botorch.acquisition import qUpperConfidenceBound

from pfgs_bo.bo.botorch_optimiser import BoTorchOptimizer
from pfgs_bo.config import PROJECT_ROOT
from pfgs_bo.experiment import (
    RunHistory,
    add_observation,
    build_datasets,
    evaluate_surrogate,
    finish_run,
    fit_surrogate,
    load_experiment,
    log_iteration,
    print_progress,
    start_run,
    unit_bounds,
)
from pfgs_bo.model import GPModel
from pfgs_bo.pareto_front.beta import beta_ranges
from pfgs_bo.plotting.results import plot_beta_schedule
from pfgs_bo.tracking.results_io import from_json

DEFAULT_BETA = 1.96


def main():
    config, space = load_experiment("BO-PFGS")
    lower_bound, upper_bound = unit_bounds(space)
    optimizer_parameters = config["optimizer_parameters"]

    if "beta_file" not in config["files"]:
        raise ValueError(
            "Set [files] beta_file in the config to the beta intervals saved by "
            'run_pfgs.py, e.g. beta_file = "results/pfgs/beta_seed_55.json".'
        )
    beta_schedules = beta_ranges(from_json(PROJECT_ROOT / config["files"]["beta_file"]))

    for name, beta_values in beta_schedules.items():
        train_data, test_sets = build_datasets(config, space)
        model = GPModel()
        history = RunHistory()
        current_beta = DEFAULT_BETA

        with start_run(
            config,
            space,
            run_name=f"{name}-{config['test_name']}",
            beta_schedule=name,
        ) as tracker:
            for i, beta in enumerate(beta_values):
                step = i + 1
                if np.isfinite(beta):
                    current_beta = beta
                else:
                    beta = current_beta

                best_observed = train_data.y_data.max()
                normalisation_stats = fit_surrogate(model, train_data, space)

                optimizer = BoTorchOptimizer(
                    acq_fn=qUpperConfidenceBound,
                    lower_bounds=lower_bound,
                    upper_bounds=upper_bound,
                    num_restarts=optimizer_parameters["num_restarts"],
                    raw_samples=optimizer_parameters["raw_samples"],
                    q=optimizer_parameters["q"],
                    beta=beta,
                )
                x_new, _ = optimizer.optimize(model, train_data)

                new_point = add_observation(
                    train_data, x_new, normalisation_stats, config["simulation"]
                )
                metrics = evaluate_surrogate(model, test_sets, normalisation_stats)
                history.record(best_observed, metrics, beta_value=beta)

                print_progress(step, metrics, beta=beta)
                log_iteration(
                    tracker,
                    step,
                    model,
                    train_data,
                    new_point,
                    metrics,
                    best_observed,
                    curr_beta=beta,
                )

            finish_run(
                tracker,
                model,
                train_data,
                history,
                results_dir=f"results/pfgs-bo/{name}",
                seed=config["train_seed"],
                beta_fig=plot_beta_schedule(beta_values),
            )


if __name__ == "__main__":
    main()
