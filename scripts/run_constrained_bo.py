"""
Constrained Bayesian optimisation with a UCB or EI acquisition function, optimised
with particle swarm optimisation (PSO).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pfgs_bo.bo.acquisition import compute_ei, compute_ucb
from pfgs_bo.bo.pso_optimiser import PSOObjective, PSOptimiser
from pfgs_bo.experiment import (
    RunHistory,
    add_observation,
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


def main() -> None:
    """Runs constrained BO with the config given on the command line."""
    config, space = load_experiment("constrained-BO")
    train_data, test_sets = build_datasets(config, space)
    lower_bound, upper_bound = unit_bounds(space)

    acquisition_function = config["acquisition_function"]
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

            acq_fun: Callable[..., NDArray[np.float64]]
            acq_kwargs: dict[str, Any]
            if acquisition_function == "UCB":
                acq_fun, acq_kwargs = compute_ucb, {"beta": config["beta"]}
            else:
                acq_fun, acq_kwargs = compute_ei, {"train_data": train_data}

            objective = PSOObjective(
                model=model,
                acq_fun=acq_fun,
                mean_discard_threshold=mean_discard_threshold,
                std_dev_discard_threshold=std_dev_discard_threshold,
                normalisation_stats=normalisation_stats,
                **acq_kwargs,
            )
            optimiser = PSOptimiser(
                objective=objective,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                seed=next(seeds),
            )
            _, x_new, stop_optimisation = optimiser.run()
            if stop_optimisation:
                break

            new_point = add_observation(
                train_data, x_new, normalisation_stats, config["simulation"]
            )
            metrics = evaluate_surrogate(model, test_sets, normalisation_stats)
            history.record(best_observed, metrics)

            print_progress(step, metrics)
            log_iteration(
                tracker, step, model, train_data, new_point, metrics, best_observed
            )

        finish_run(
            tracker,
            model,
            train_data,
            history,
            csv_path=results_csv(config),
        )


if __name__ == "__main__":
    main()
