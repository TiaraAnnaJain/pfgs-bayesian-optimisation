"""MLflow tracker that logs parameters, metrics and figures for each run."""

import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from numpy.typing import NDArray

from pfgs_bo.data.investigation_space import InvestigationSpaceVariable

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def tracking_uri() -> str:
    """MLFLOW_TRACKING_URI if set, else mlflow.db in the project root."""
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT}/mlflow.db")


def experiment_name(problem: str) -> str:
    """MLflow experiment for a problem label, e.g. "himmelblau-optimisation"."""
    return f"{problem}-optimisation"


def prepare_experiments(problems: list[str]) -> None:
    """
    Creates the MLflow database and an experiment per problem, if they don't exist.

    Parameters:
        problems (list[str]): problem labels, e.g. ["himmelblau", "ackley_6d"].
    """
    mlflow.set_tracking_uri(tracking_uri())
    for problem in problems:
        if mlflow.get_experiment_by_name(experiment_name(problem)) is None:
            mlflow.create_experiment(experiment_name(problem))


class MLflowTracker:
    """Logs parameters, model metrics and plots to MLflow."""

    def __init__(self, experiment_name: str, run_name: str | None = None):
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        print(mlflow.get_tracking_uri())

    def log_resolved_config(self, resolved_config: dict[str, Any]):
        """
        Log only parameters used in the run from config file.

        Parameters:
            resolved_config (dict[str, Any]): fully resolved config.
        """
        config_to_log = deepcopy(resolved_config)
        for key in [
            "benchmark",
            "problem",
            "simulation",
            "train_samples",
            "test_samples",
            "test_samples_per_region",
            "delta",
            "maxima",
            "train_seed",
        ]:
            config_to_log.pop(key, None)

        params = self._flatten(config_to_log)
        mlflow.log_params(params)

    @staticmethod
    def _flatten(d: dict, parent_key: str = "") -> dict:
        """Recursively flattens nested dicts using dot notation as keys."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}: {k}" if parent_key else k
            if isinstance(v, dict):
                items.update(MLflowTracker._flatten(v, new_key))
            else:
                items[new_key] = v
        return items

    def log_experiment_setup(
        self,
        x_label: list[str],
        y_label: list[str],
        variables: list[InvestigationSpaceVariable],
        problem: str,
        beta_schedule: str | None = None,
        train_seed: int | None = None,
    ):
        """
        Logs experiment setup as parameters.

        Parameters:
            x_label (list[str]): input labels.
            y_label (list[str]): output labels.
            variables (list[InvestigationSpaceVariable]): variables to be logged.
            problem (str): name of the benchmark problem.
            beta_schedule (str): Name of the beta schedule.
            train_seed (int): seed for training runs.
        """
        bounds = {
            f"{v.label}_bounds": [v.lower_bound, v.upper_bound] for v in variables
        }

        mlflow.log_params(
            {
                "x_labels": x_label,
                "y_labels": y_label,
                "problem": problem,
                **bounds,
                "beta_schedule": beta_schedule,
                "train_seed": train_seed,
            }
        )

    def log_iteration(
        self,
        step: int,
        entire_space_rrmse: float,
        design_space_rrmse: float,
        train_data_max: float,
        gp_fig: plt.Figure | None = None,
        pareto_fig: plt.Figure | None = None,
        beta_min: float | None = None,
        beta_max: float | None = None,
        adjusted_thresholds: NDArray[np.float64] | None = None,
        curr_beta: float | None = None,
    ):
        """
        Logs metrics per iteration.

        Parameters:
            step (int): current iteration.
            entire_space_rrmse (float): RRMSE on the entire-space test set.
            design_space_rrmse (float): RRMSE on the design-space test set.
            train_data_max (float): maximum training data value.
            gp_fig (plt.Figure | None): GP contour plot at current iteration.
            pareto_fig (plt.Figure | None): Pareto front plot at current iteration.
            beta_min (float | None): lower bound of the beta range.
            beta_max (float | None): upper bound of the beta range.
            adjusted_thresholds (NDArray[np.float64] | None): adjusted discard
                thresholds on pareto front.
            curr_beta (float | None): current beta value.
        """
        mlflow.log_metrics(
            {"entire_space_rrmse": round(entire_space_rrmse, 4)}, step=step
        )
        mlflow.log_metrics(
            {"design_space_rrmse": round(design_space_rrmse, 4)}, step=step
        )
        mlflow.log_metrics({"train_data_max": round(train_data_max, 4)}, step=step)

        if gp_fig is not None:
            self._log_fig(gp_fig, f"gp_contour/iter_{step:03d}.png")

        beta_bounds = {"beta_min": beta_min, "beta_max": beta_max}
        finite_bounds = {
            name: value
            for name, value in beta_bounds.items()
            if value is not None and np.isfinite(value)
        }
        if finite_bounds:
            mlflow.log_metrics(finite_bounds, step=step)

        if adjusted_thresholds is not None:
            mlflow.log_metrics(
                {
                    "adjusted_thresholds_mean": round(float(adjusted_thresholds[0]), 2),
                    "adjusted_thresholds_std_dev": round(
                        float(adjusted_thresholds[1]), 2
                    ),
                },
                step=step,
            )

        if curr_beta is not None:
            mlflow.log_metrics({"beta": curr_beta}, step=step)

        if pareto_fig is not None:
            self._log_fig(pareto_fig, f"pareto_fronts/iter_{step:03d}.png")

    def log_final_results(
        self,
        final_gp_fig: plt.Figure,
        rrmse_fig: plt.Figure,
        beta_fig: plt.Figure | None = None,
        train_data_max_fig: plt.Figure | None = None,
    ):
        """
        Logs final results and figures.

        Parameters:
            final_gp_fig (plt.Figure): Final GP contour plot.
            rrmse_fig (plt.Figure): Final RRMSE figure.
            beta_fig (plt.Figure | None): Final beta figure with feasible ranges.
            train_data_max_fig (plt.Figure | None): best observed value per
                iteration.
        """
        self._log_fig(final_gp_fig, "final/gp_contour_final.png")
        self._log_fig(rrmse_fig, "final/rrmse_plot.png")

        if beta_fig is not None:
            self._log_fig(beta_fig, "final/beta_plot.png")

        if train_data_max_fig is not None:
            self._log_fig(train_data_max_fig, "final/train_data_max_final.png")

    def _log_fig(self, fig: plt.Figure, artifact_path: str):
        """
        Logs plots into specified MLflow directory.

        Parameters:
            fig (plt.Figure): figure to log.
            artifact_path (str): artifact path.
        """
        path = Path(artifact_path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_file = Path(tmp_dir) / path.name
            fig.savefig(local_file, bbox_inches="tight", dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(local_file), artifact_path=str(path.parent))

    def finish(self, failed: bool = False) -> None:
        """Ends the MLflow run, marking it FAILED or FINISHED."""
        mlflow.end_run(status="FAILED" if failed else "FINISHED")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.finish(failed=exc_type is not None)
