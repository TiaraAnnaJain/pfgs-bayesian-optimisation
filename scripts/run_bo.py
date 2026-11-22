"""
Standard Bayesian optimisation with a UCB or EI acquisition function, optimised
with BoTorch's gradient-based optimiser.
"""

from typing import Any

from botorch.acquisition import (
    AcquisitionFunction,
    qExpectedImprovement,
    qUpperConfidenceBound,
)

from pfgs_bo.bo.botorch_optimiser import BoTorchOptimizer
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
    results_csv,
    start_run,
    unit_bounds,
)
from pfgs_bo.model import GPModel


def main() -> None:
    """Runs standard BO with the config given on the command line."""
    config, space = load_experiment("BO")
    train_data, test_sets = build_datasets(config, space)
    lower_bound, upper_bound = unit_bounds(space)

    acquisition_function = config["acquisition_function"]
    acq_fn: type[AcquisitionFunction]
    acq_kwargs: dict[str, Any]
    if acquisition_function == "UCB":
        acq_fn, acq_kwargs = qUpperConfidenceBound, {"beta": config["beta"]}
    else:
        acq_fn, acq_kwargs = qExpectedImprovement, {}

    optimizer = BoTorchOptimizer(
        acq_fn=acq_fn,
        lower_bounds=lower_bound,
        upper_bounds=upper_bound,
        num_restarts=config["optimizer_parameters"]["num_restarts"],
        raw_samples=config["optimizer_parameters"]["raw_samples"],
        q=config["optimizer_parameters"]["q"],
        **acq_kwargs,
    )

    model = GPModel()
    history = RunHistory()

    with start_run(config, space) as tracker:
        for i in range(config["num_iter"]):
            step = i + 1
            best_observed = train_data.y_data.max()
            normalisation_stats = fit_surrogate(model, train_data, space)

            # Maximise the acquisition function to choose the next point.
            x_new, _ = optimizer.optimize(model, train_data)

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
