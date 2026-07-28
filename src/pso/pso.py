"""Particle swarm optimization objective and optimizer."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import pyswarms as ps

from data.data_models import Dataset
from pso.filtered_acquisition import filter_predictions


class PSOObjective:
    """
    Defining the objective function to be optimized by parameter swarm optimization.

    Parameters:
        model (Any): GP model.
        acq_fun (Callable): Acquisition function.
        acq_kwargs: Extra keyword arguments passed to the acquisition function.
        mean_discard_threshold (Optional[NDArray[np.float64]]): mean discard threshold.
        std_dev_discard_threshold (Optional[NDArray[np.float64]]): standard deviation
            discard threshold.
        normalisation_stats (Optional[Dict[str, Any]]): normalisation stats from train data.
    """

    def __init__(
        self,
        model: Any,
        acq_fun: Callable,
        device="cpu",
        mean_discard_threshold: Optional[NDArray[np.float64]] = None,
        std_dev_discard_threshold: Optional[NDArray[np.float64]] = None,
        normalisation_stats: Optional[Dict[str, Any]] = None,
        **acq_kwargs,
    ):
        self.model = model
        self.x_labels = model.x_labels
        self.acq_fun = acq_fun
        self.device = device
        self.mean_discard_threshold = mean_discard_threshold
        self.std_dev_discard_threshold = std_dev_discard_threshold
        self.acq_kwargs = acq_kwargs
        self.normalisation_stats = normalisation_stats
        self.stop_optimisation = False

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
            dataset_type="recipe",
            x_data=x,
            x_labels=self.x_labels,
            normalised=True,
        )
        candidate_dataset.check()

        predictions = self.model.predict(candidate_dataset)
        costs = -np.asarray(
            self.acq_fun(predictions, **self.acq_kwargs), dtype=np.float64
        ).reshape(-1)

        if (
            self.mean_discard_threshold is not None
            and self.std_dev_discard_threshold is not None
        ):
            mask = filter_predictions(
                predictions,
                self.mean_discard_threshold,
                self.std_dev_discard_threshold,
                self.normalisation_stats,
            )

            if not np.any(mask):
                self.stop_optimisation = True
                return np.full_like(costs, np.inf)

            costs[~mask] = np.inf

        return costs


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
        bounds = (xl, xu)

        if n_particles is None:
            n_particles = 200
        if options is None:
            options = {"c1": 2.0, "c2": 2.0, "w": 0.9}

        self.objective = objective
        init_pos = np.random.uniform(0, 1, size=(n_particles, len(xl)))
        self.optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(xl),
            options=options,
            bounds=bounds,
            init_pos=init_pos,
        )

    def run(
        self,
    ) -> Tuple[
        Union[None, NDArray[np.float64]], Union[None, NDArray[np.float64]], bool
    ]:
        """
        Runs the particle swarm optimization. Stops optimization loop if no point survives thresholds.

        Returns:
            cost (NDArray[np.float64]): value of PSO at points.
            positions (NDArray[np.float64]): positions of particles.
            stop_optimisation (bool): Whether optimization should stop due to no feasible points.
        """

        initial_costs = self.objective(self.optimizer.swarm.position)

        if self.objective.stop_optimisation:
            return None, None, True

        cost, best_pos = self.optimizer.optimize(self.objective, iters=100)

        return cost, best_pos, False
