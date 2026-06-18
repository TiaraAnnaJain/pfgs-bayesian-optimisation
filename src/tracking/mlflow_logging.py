"""Contains MLFlow tracking class to log parameters, metrics and artifacts into MLflow."""

import tempfile
import os
from typing import List, Optional, Any, Dict
from pathlib import Path

import mlflow
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from src.data.investigation_space import InvestigationSpaceVariable

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
        params = self._flatten(resolved_config)
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
        simulation_name: str,
        variables: List[InvestigationSpaceVariable],
        mean_discard_threshold: Optional[NDArray[np.float64]] = None,
        variance_discard_threshold: Optional[NDArray[np.float64]] = None,
    ):
        """
        Logs experiment setup as parameters.

        Parameters:
            x_label (List[str]): input labels.
            y_label (List[str]): output labels.
            simulation_name (str): simulation being run.
            variables (List[InvestigationSpaceVariable]): variables to be logged.
            mean_discard_threshold (Optional[NDArray[np.float64]]): mean discard
                threshold on pareto front.
            variance_discard_threshold (Optional[NDArray[np.float64]]): variance
                discard threshold on pareto front.
        """
        bounds = {
            f"{v.label}_bounds": [v.lower_bound, v.upper_bound] for v in variables
        }

        mlflow.log_params(
            {
                "x_labels": x_label,
                "y_labels": y_label,
                "simulation_name": simulation_name,
                **bounds,
            }
        )

        if (
            mean_discard_threshold is not None
            and variance_discard_threshold is not None
        ):
            mlflow.log_params(
                {
                    "discard_thresholds": [
                        mean_discard_threshold.item(),
                        variance_discard_threshold.item(),
                    ],
                }
            )

    def log_iteration(
        self,
        step: int,
        rrmse: float,
        gp_fig: Optional[plt.Figure] = None,
        pareto_fig: Optional[plt.Figure] = None,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        adjusted_thresholds: Optional[NDArray[np.float64]] = None,
    ):
        """
        Logs metrics per iteration.

        Parameters:
            step (int): current iteration.
            rrmse (float): RRMSE value at current iteration.
            gp_fig (Optional[plt.Figure]): GP contour plot at current iteration.
            pareto_fig (Optional[plt.Figure]): Pareto front plot at current iteration.
            alpha_min (Optional[float]): lower bound of the alpha range.
            alpha_max (Optional[float]): upper bound of the alpha range.
            adjusted_thresholds (Optional[NDArray[np.float64]]): adjusted discard
                thresholds on pareto front.
        """
        mlflow.log_metrics({"rrmse": round(rrmse, 4)}, step=step)

        if gp_fig is not None:
            self._log_fig(gp_fig, f"gp_contour/iter_{step:03d}.png")

        if alpha_min is not None and alpha_max is not None:
            if np.isfinite(alpha_min) and np.isfinite(alpha_max):
                mlflow.log_metrics(
                    {
                        "alpha_min": alpha_min,
                        "alpha_max": alpha_max,
                    },
                    step=step,
                )
            else:
                mlflow.log_metrics(
                    {
                        "alpha_min": 0,
                        "alpha_max": 0,
                    },
                    step=step,
                )

        if adjusted_thresholds is not None:
            mlflow.log_metrics(
                {
                    "adjusted_thresholds_mean": round(float(adjusted_thresholds[0]), 2),
                    "adjusted_thresholds_variance": round(
                        float(adjusted_thresholds[1]), 2
                    ),
                },
                step=step,
            )

        if pareto_fig is not None:
            self._log_fig(pareto_fig, f"pareto_fronts/iter_{step:03d}.png")

    def log_final_results(
        self,
        final_gp_fig: plt.Figure,
        rrmse_fig: plt.Figure,
        alpha_fig: Optional[plt.Figure] = None,
    ):
        """
        Logs final results and figures.

        Parameters:
            final_gp_fig (plt.Figure): Final GP contour plot.
            rrmse_fig (plt.Figure): Final RRMSE figure.
            alpha_fig (Optional[plt.Figure]): Final alpha figure with feasible ranges.
        """
        self._log_fig(final_gp_fig, "final/gp_contour_final.png")
        self._log_fig(rrmse_fig, "final/rrmse_plot.png")

        if alpha_fig is not None:
            self._log_fig(alpha_fig, "final/alpha_plot.png")

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
