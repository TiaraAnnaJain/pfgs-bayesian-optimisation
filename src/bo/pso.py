"""Particle swarm optimization objective and optimizer."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import pyswarms as ps

from src.data.data_models import Dataset


class PSOObjective:
    """
    Defining the objective function to be optimized by parameter swarm optimization.

    Parameters:
        model (Any): GP model.
        acq_fun (Callable): Acquisition function.
        acq_kwargs: Extra keyword arguments passed to the acquisition function.
            """
    def __init__(self, model: Any, acq_fun: Callable, device="cpu", **acq_kwargs):
        self.model = model
        self.x_labels = model.x_labels
        self.acq_fun = acq_fun
        self.device = device
        self.acq_kwargs = acq_kwargs

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluates GP and computes acquisition function for given predictions, the negates
        acquisition function outputs.

        Parameters:
            x (NDArray[np.float64]): Population points to be evaluated.

        Returns:
            acq_values (NDArray[np.float64]): negated acquisition function values.
        """
        candidate_dataset = Dataset(
            dataset_type="recipe", x_data=x, x_labels=self.x_labels, normalised=True
        )
        candidate_dataset.check()

        predictions = self.model.predict(candidate_dataset)
        acq_values = self.acq_fun(predictions, **self.acq_kwargs)

        return -acq_values


class PSOptimiser:
    """
    PSO Optimiser.

    Parameters:
        objective (PSOObjective): Objective function to be optimized.
        lower_bound (List[float]): Lower bound of the optimization space.
        upper_bound (List[float]): Upper bound of the optimization space.
        n_particles (Optional[int]): Number of particles, defaults to 200.
        options (Optional[Dict[str, float]]): hyperparameter values. Defaults to
            {"c1": 2.0, "c2": 2.0, "w": 0.9}
    """

    def __init__(
        self,
        objective: PSOObjective,
        lower_bound: List[float],
        upper_bound: List[float],
        n_particles: Optional[int] = None,
        options: Optional[Dict[str, float]] = None,
    ):
        xl = np.array(lower_bound, dtype=np.float64)
        xu = np.array(upper_bound, dtype=np.float64)
        bounds = [xl, xu]

        if n_particles is None:
            n_particles = 200
        if options is None:
            options = {"c1": 2.0, "c2": 2.0, "w": 0.9}

        self.objective = objective
        self.optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(xl),
            options=options,
            bounds=bounds,
        )

    def run(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Runs the particle swarm optimization.

        Returns:
            cost (NDArray[np.float64]): value of PSO at points.
            positions (NDArray[np.float64]): positions of particles.
        """
        cost, best_pos = self.optimizer.optimize(self.objective, iters=100)
        return cost, best_pos
