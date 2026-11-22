"""Shared building blocks for the experiment scripts."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from pfgs_bo.config import (
    PROJECT_ROOT,
    load_config,
    resolve_config,
    validate_config,
)
from pfgs_bo.data.data_models import Dataset
from pfgs_bo.data.data_processing import generate_data, generate_design_space_dataset
from pfgs_bo.data.investigation_space import InvestigationSpace
from pfgs_bo.metrics import calculate_model_metrics
from pfgs_bo.model import GPModel
from pfgs_bo.plotting.gp import contour_slice_plot
from pfgs_bo.plotting.results import plot_rrmse_history, plot_train_max_history
from pfgs_bo.tracking.mlflow_tracker import MLflowTracker
from pfgs_bo.tracking.results_io import to_csv

TestSets = dict[str, Dataset]


def load_experiment(
    method: str, argv: Sequence[str] | None = None
) -> tuple[dict[str, Any], InvestigationSpace]:
    """
    Loads and validates the config given on the command line, and the search space
    of its problem.

    Parameters:
        method (str): method the calling script runs; the config's ``method`` must
            match, so a config can't be run with the wrong script.
        argv (Sequence[str] | None): command-line arguments; defaults to
            ``sys.argv[1:]``.

    Returns:
        config (dict[str, Any]): resolved config (see ``resolve_config``).
        space (InvestigationSpace): search space of the config's problem.
    """
    config = load_config(argv)
    validate_config(config)
    if config["method"] != method:
        raise ValueError(
            f"This script runs method {method!r}, but the config has method "
            f"{config['method']!r}."
        )

    resolved = resolve_config(config)

    return resolved, resolved["benchmark"].space


def _number(value: float) -> str:
    """Formats a number for a folder name: 1000.0 -> "1000", 0.5 -> "0.5"."""
    return f"{value:g}"


def _thresholds_tag(config: dict[str, Any]) -> str:
    """Folder name for a pair of discard thresholds, e.g. "mean1000_std50"."""
    mean = _number(config["mean_discard_threshold"])
    std = _number(config["std_dev_discard_threshold"])
    return f"mean{mean}_std{std}"


def _acquisition_tag(config: dict[str, Any]) -> str:
    """Folder name for an acquisition function, e.g. "EI" or "UCB_beta1.96"."""
    if config["acquisition_function"] == "UCB":
        return f"UCB_beta{_number(config['beta'])}"
    return config["acquisition_function"]


def run_dir(config: dict[str, Any], schedule: str | None = None) -> Path:
    """
    Returns the results folder for a run, derived from its config.

    Runs that differ in problem, method or method settings get different folders,
    so sweeps don't overwrite each other::

        results/<problem>/pfgs/mean<m>_std<s>/
        results/<problem>/bo/<UCB_beta<b> | EI>/
        results/<problem>/pfgs-bo/beta_from_mean<m>_std<s>/<schedule>/
        results/<problem>/constrained-bo/<UCB_beta<b> | EI>/mean<m>_std<s>/

    For PFGS-guided BO the thresholds are those of the PFGS run whose beta
    intervals are used. The results root can be changed with ``results_dir``.

    Parameters:
        config (dict[str, Any]): resolved config.
        schedule (str | None): beta schedule name, for PFGS-guided BO.

    Returns:
        path (Path): results folder.
    """
    root = PROJECT_ROOT / config.get("results_dir", "results") / config["problem"]
    method = config["method"]

    if method == "PFGS":
        return root / "pfgs" / _thresholds_tag(config)
    if method == "BO":
        return root / "bo" / _acquisition_tag(config)
    if method == "BO-PFGS":
        if schedule is None:
            raise ValueError("PFGS-guided BO results need a beta schedule name.")
        return root / "pfgs-bo" / f"beta_from_{_thresholds_tag(config)}" / schedule
    if method == "constrained-BO":
        return (
            root / "constrained-bo" / _acquisition_tag(config) / _thresholds_tag(config)
        )
    raise ValueError(f"Method {method} is not supported.")


def results_csv(config: dict[str, Any], schedule: str | None = None) -> Path:
    """
    Returns the CSV file for a run: ``train_seed_<n>.csv`` in its ``run_dir``.

    Parameters:
        config (dict[str, Any]): resolved config.
        schedule (str | None): beta schedule name, for PFGS-guided BO.

    Returns:
        path (Path): CSV file.
    """
    return run_dir(config, schedule) / f"train_seed_{config['train_seed']}.csv"


def beta_file(config: dict[str, Any]) -> Path:
    """
    Returns the beta-interval file of the PFGS run matching a config's problem,
    discard thresholds and training seed.

    PFGS writes this file and PFGS-guided BO reads it, so both resolve to the same
    path.

    Parameters:
        config (dict[str, Any]): resolved PFGS or PFGS-guided BO config.

    Returns:
        path (Path): ``beta_train_seed_<n>.json`` in the PFGS run's folder.
    """
    pfgs_dir = run_dir({**config, "method": "PFGS"})
    return pfgs_dir / f"beta_train_seed_{config['train_seed']}.json"


def unit_bounds(space: InvestigationSpace) -> tuple[list[float], list[float]]:
    """
    Returns the bounds of the normalised search space, [0, 1] in every dimension.

    Parameters:
        space (InvestigationSpace): investigation space.

    Returns:
        lower_bound (list[float]): lower bounds (all 0).
        upper_bound (list[float]): upper bounds (all 1).
    """
    n_vars = len(space.labels)
    return [0.0] * n_vars, [1.0] * n_vars


def build_datasets(
    config: dict[str, Any], space: InvestigationSpace
) -> tuple[Dataset, TestSets]:
    """
    Generates the initial training set and the two test sets.

    Parameters:
        config (dict[str, Any]): resolved config.
        space (InvestigationSpace): investigation space.

    Returns:
        train_data (Dataset): LHS training set over the entire investigation space.
        test_sets (TestSets): "Entire Space" and "Design Space" test sets.
    """
    simulation = config["simulation"]
    maxima = np.array(config["benchmark"].maxima)
    x_labels = space.labels
    y_labels = space.outputs

    train_data = generate_data(
        simulation=simulation,
        n_samples=config["train_samples"],
        n_vars=len(x_labels),
        lower_bound=space.lower_bound_list,
        upper_bound=space.upper_bound_list,
        seed=config["train_seed"],
        dataset_type="train",
        x_labels=x_labels,
        y_labels=y_labels,
    )
    entire_space = generate_data(
        simulation=simulation,
        n_samples=config["test_samples"],
        n_vars=len(x_labels),
        lower_bound=space.lower_bound_list,
        upper_bound=space.upper_bound_list,
        seed=config["test_seed"],
        dataset_type="test",
        x_labels=x_labels,
        y_labels=y_labels,
    )
    design_space = generate_design_space_dataset(
        simulation=simulation,
        n_samples=config["test_samples"],
        n_vars=len(x_labels),
        delta=config["delta"],
        seed=config["test_seed"],
        dataset_type="test",
        x_labels=x_labels,
        y_labels=y_labels,
        maxima=maxima,
    )

    return train_data, {"Entire Space": entire_space, "Design Space": design_space}


def start_run(
    config: dict[str, Any],
    space: InvestigationSpace,
    run_name: str | None = None,
    beta_schedule: str | None = None,
) -> MLflowTracker:
    """
    Starts an MLflow run and logs the config and experiment setup. Use the returned
    tracker as a context manager so the run is always closed.

    Parameters:
        config (dict[str, Any]): resolved config.
        space (InvestigationSpace): investigation space.
        run_name (str | None): MLflow run name, defaults to the config's test name.
        beta_schedule (str | None): name of the beta schedule, for PFGS-guided BO.

    Returns:
        tracker (MLflowTracker): tracker for the started run.
    """
    problem = config["problem"]
    tracker = MLflowTracker(
        experiment_name=f"{problem}-optimisation",
        run_name=run_name or config["test_name"],
    )

    try:
        tracker.log_resolved_config(config)
        tracker.log_experiment_setup(
            x_label=space.labels,
            y_label=space.outputs,
            variables=space.variables,
            problem=problem,
            beta_schedule=beta_schedule,
            train_seed=config["train_seed"],
        )
    except Exception:
        tracker.finish(failed=True)
        raise

    return tracker


def fit_surrogate(
    model: GPModel, train_data: Dataset, space: InvestigationSpace
) -> dict[str, dict]:
    """
    Normalises the training data and fits the GP surrogate to it. The training data
    is left normalised; "add_observation" de-normalises it again.

    Parameters:
        model (GPModel): GP surrogate to fit.
        train_data (Dataset): de-normalised training data.
        space (InvestigationSpace): investigation space, used as normalisation bounds.

    Returns:
        normalisation_stats (dict[str, Dict]): statistics used to normalise the data.
    """
    train_data.add_normalisation_params(
        lower_bound=space.lower_bounds, upper_bound=space.upper_bounds
    )
    train_data.normalise()
    normalisation_stats = train_data.get_stats()

    model.train(train_data)

    return normalisation_stats


def add_observation(
    train_data: Dataset,
    x_new: ArrayLike,
    normalisation_stats: dict[str, dict],
    simulation: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> Dataset:
    """
    Evaluates the simulation at a new normalised point and adds it to the training data.

    Parameters:
        train_data (Dataset): normalised training data; it is de-normalised in place.
        x_new (ArrayLike): new point in the normalised search space.
        normalisation_stats (dict[str, Dict]): statistics used to normalise the data.
        simulation (Callable): objective function to evaluate.

    Returns:
        x_new (Dataset): the new point, de-normalised.
    """
    train_data.de_normalise()

    new_point = Dataset(
        dataset_type="recipe",
        x_labels=train_data.x_labels,
        x_data=np.asarray(x_new).reshape(1, -1),
        normalised=True,
    )
    new_point.set_stats(normalisation_stats)
    new_point.de_normalise()

    y_new = simulation(new_point.x_data).reshape(1, 1)

    train_data.x_data = np.vstack([train_data.x_data, new_point.x_data])
    train_data.y_data = np.vstack([train_data.y_data, y_new])

    return new_point


def evaluate_surrogate(
    model: GPModel, test_sets: TestSets, normalisation_stats: dict[str, dict]
) -> dict[str, float]:
    """
    Computes the RRMSE of the surrogate on both test sets.

    Parameters:
        model (GPModel): fitted GP surrogate.
        test_sets (TestSets): "Entire Space" and "Design Space" test sets.
        normalisation_stats (dict[str, Dict]): statistics used to normalise the data.

    Returns:
        metrics (dict[str, float]): "entire_space_rrmse" and "design_space_rrmse".
    """
    metrics = calculate_model_metrics(model, test_sets, normalisation_stats)

    return {
        "entire_space_rrmse": metrics["Entire Space RRMSE"],
        "design_space_rrmse": metrics["Design Space RRMSE"],
    }


@dataclass
class RunHistory:
    """
    Per-iteration results of a single run.

    Parameters:
        entire_space_rrmse (list[float]): RRMSE on the entire-space test set.
        design_space_rrmse (list[float]): RRMSE on the design-space test set.
        train_data_max (list[float]): best observed value before each new point.
        extra (dict[str, list[float]]): method-specific values, e.g. beta.
    """

    entire_space_rrmse: list[float] = field(default_factory=list)
    design_space_rrmse: list[float] = field(default_factory=list)
    train_data_max: list[float] = field(default_factory=list)
    extra: dict[str, list[float]] = field(default_factory=dict)

    def record(
        self, train_data_max: float, metrics: dict[str, float], **extra: float
    ) -> None:
        """
        Records the results of one iteration.

        Parameters:
            train_data_max (float): best observed value before the new point was added.
            metrics (dict[str, float]): output of "evaluate_surrogate".
            **extra (float): method-specific values to record, e.g. beta.
        """
        self.train_data_max.append(train_data_max)
        self.entire_space_rrmse.append(metrics["entire_space_rrmse"])
        self.design_space_rrmse.append(metrics["design_space_rrmse"])
        for name, value in extra.items():
            self.extra.setdefault(name, []).append(value)

    def to_dict(self) -> dict[str, Any]:
        """Returns the history as columns, ready to be written to CSV."""
        return {
            "iteration": range(len(self.entire_space_rrmse)),
            "entire_space_rrmse": self.entire_space_rrmse,
            "design_space_rrmse": self.design_space_rrmse,
            "train_data_max": self.train_data_max,
            **self.extra,
        }


def print_progress(step: int, metrics: dict[str, float], **extra: Any) -> None:
    """
    Prints a one-line summary of an iteration.

    Parameters:
        step (int): iteration number, starting at 1.
        metrics (dict[str, float]): output of "evaluate_surrogate".
        **extra (Any): method-specific values to print, e.g. beta.
    """
    values = {**metrics, **extra}
    print(f"Iteration {step}: " + ", ".join(f"{k} = {v}" for k, v in values.items()))


def log_iteration(
    tracker: MLflowTracker,
    step: int,
    model: GPModel,
    train_data: Dataset,
    new_point: Dataset,
    metrics: dict[str, float],
    best_observed: float,
    **extra: Any,
) -> None:
    """
    Logs the metrics and GP contour plot of an iteration to MLflow.

    Parameters:
        tracker (MLflowTracker): active tracker.
        step (int): iteration number, starting at 1.
        model (GPModel): fitted GP surrogate.
        train_data (Dataset): de-normalised training data, including the new point.
        new_point (Dataset): the point added at this iteration.
        metrics (dict[str, float]): output of "evaluate_surrogate".
        **extra (Any): method-specific arguments for "MLflowTracker.log_iteration".
    """
    gp_fig = contour_slice_plot(
        trained_model=model,
        train_dataset=train_data,
        x_plot_labels=train_data.x_labels[:2],
        y_plot_label=train_data.y_labels[0],
        iteration=step,
        x_new=new_point,
    )

    tracker.log_iteration(
        step=step,
        entire_space_rrmse=metrics["entire_space_rrmse"],
        design_space_rrmse=metrics["design_space_rrmse"],
        train_data_max=best_observed,
        gp_fig=gp_fig,
        **extra,
    )


def finish_run(
    tracker: MLflowTracker,
    model: GPModel,
    train_data: Dataset,
    history: RunHistory,
    csv_path: Path,
    beta_fig: plt.Figure | None = None,
) -> None:
    """
    Logs the end-of-run figures to MLflow and writes the run history to CSV.

    Parameters:
        tracker (MLflowTracker): active tracker.
        model (GPModel): final GP surrogate.
        train_data (Dataset): final de-normalised training data.
        history (RunHistory): per-iteration results.
        csv_path (Path): CSV file to write the history to (see ``results_csv``).
        beta_fig (plt.Figure | None): beta figure, for PFGS and PFGS-guided BO.
    """
    rrmse_fig = plot_rrmse_history(
        history.entire_space_rrmse, history.design_space_rrmse
    )
    train_data_max_fig = plot_train_max_history(history.train_data_max)
    final_gp_fig = contour_slice_plot(
        trained_model=model,
        train_dataset=train_data,
        x_plot_labels=train_data.x_labels[:2],
        y_plot_label=train_data.y_labels[0],
        iteration="Final",
    )

    tracker.log_final_results(
        final_gp_fig=final_gp_fig,
        rrmse_fig=rrmse_fig,
        beta_fig=beta_fig,
        train_data_max_fig=train_data_max_fig,
    )

    to_csv(results=history.to_dict(), csv_file=csv_path)
