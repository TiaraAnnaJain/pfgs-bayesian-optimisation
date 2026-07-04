"""Contains MLFlow tracking class to log parameters, metrics and artifacts into MLflow."""

import os
from copy import deepcopy
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from numpy.typing import NDArray

from data.investigation_space import InvestigationSpaceVariable

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MLflowTracker:
    """Logs parameters, model metrics and plots to MLflow."""

    def __init__(self, experiment_name: str, run_name: str = None):
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        self._artifact_dir = tempfile.mkdtemp()
        print(mlflow.get_tracking_uri())

    def log_resolved_config(self, resolved_config: Dict[str, Any]):
        """
        Log only parameters used in the run from config file.

        Parameters:
            resolved_config (Dict[str, Any]): fully resolved config.
        """
        config_to_log = deepcopy(resolved_config)
        for key in [
            "simulation",
            "train_samples",
            "test_samples",
            "test_samples_per_region",
            "delta",
            "maxima",
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
        x_label: List[str],
        y_label: List[str],
        variables: List[InvestigationSpaceVariable],
        simulation: str,
    ):
        """
        Logs experiment setup as parameters.

        Parameters:
            x_label (List[str]): input labels.
            y_label (List[str]): output labels.
            variables (List[InvestigationSpaceVariable]): variables to be logged.
            simulation (str): Name of the simulation being run.
        """
        bounds = {
            f"{v.label}_bounds": [v.lower_bound, v.upper_bound] for v in variables
        }

        mlflow.log_params(
            {
                "x_labels": x_label,
                "y_labels": y_label,
                "simulation": simulation,
                **bounds,
            }
        )

    def log_iteration(
        self,
        step: int,
        rrmse: float,
        train_data_max: float,
        gp_fig: Optional[plt.Figure] = None,
        pareto_fig: Optional[plt.Figure] = None,
        beta_min: Optional[float] = None,
        beta_max: Optional[float] = None,
        adjusted_thresholds: Optional[NDArray[np.float64]] = None,
        curr_beta: Optional[float] = None,
    ):
        """
        Logs metrics per iteration.

        Parameters:
            step (int): current iteration.
            rrmse (float): RRMSE value at current iteration.
            train_data_max (float): maximum training data value.
            gp_fig (Optional[plt.Figure]): GP contour plot at current iteration.
            pareto_fig (Optional[plt.Figure]): Pareto front plot at current iteration.
            beta_min (Optional[float]): lower bound of the beta range.
            beta_max (Optional[float]): upper bound of the beta range.
            adjusted_thresholds (Optional[NDArray[np.float64]]): adjusted discard
                thresholds on pareto front.
            curr_beta (Optional[float]): current beta value.
        """
        mlflow.log_metrics({"rrmse": round(rrmse, 4)}, step=step)
        mlflow.log_metrics({"train_data_max": round(train_data_max, 4)}, step=step)

        if gp_fig is not None:
            self._log_fig(gp_fig, f"gp_contour/iter_{step:03d}.png")

        if beta_min is not None and beta_max is not None:
            if np.isfinite(beta_min) and np.isfinite(beta_max):
                mlflow.log_metrics(
                    {
                        "beta_min": beta_min,
                        "beta_max": beta_max,
                    },
                    step=step,
                )
            else:
                mlflow.log_metrics(
                    {
                        "beta_min": 0,
                        "beta_max": 0,
                    },
                    step=step,
                )

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
        beta_fig: Optional[plt.Figure] = None,
        train_data_max_fig: Optional[plt.Figure] = None,
    ):
        """
        Logs final results and figures.

        Parameters:
            final_gp_fig (plt.Figure): Final GP contour plot.
            rrmse_fig (plt.Figure): Final RRMSE figure.
            beta_fig (Optional[plt.Figure]): Final beta figure with feasible ranges.
            train_data_max_fig (Optional[plt.Figure]): max value of train y at each iteration.
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
        full_path = os.path.join(self._artifact_dir, artifact_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        fig.savefig(full_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        folder = str(Path(artifact_path).parent)
        mlflow.log_artifact(full_path, artifact_path=folder)

    def finish(self, failed=False):
        mlflow.end_run(status="FAILED" if failed else "FINISHED")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.finish(failed=exc_type is not None)
