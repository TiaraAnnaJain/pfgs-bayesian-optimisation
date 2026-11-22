"""Particle swarm optimization objective and optimizer."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pyswarms as ps
from numpy.typing import NDArray

from pfgs_bo.bo.acquisition import filter_predictions
from pfgs_bo.data.data_models import Dataset


class PSOObjective:
    """
    Defining the objective function to be optimized by parameter swarm optimization.

    Parameters:
        model (Any): GP model.
        acq_fun (Callable): Acquisition function.
        acq_kwargs: Extra keyword arguments passed to the acquisition function.
        mean_discard_threshold (NDArray[np.float64] | None): mean discard threshold.
        std_dev_discard_threshold (NDArray[np.float64] | None): standard deviation
            discard threshold.
        normalisation_stats (dict[str, Any] | None): normalisation stats from the
            training data.
    """

    def __init__(
        self,
        model: Any,
        acq_fun: Callable,
        device="cpu",
        mean_discard_threshold: NDArray[np.float64] | None = None,
        std_dev_discard_threshold: NDArray[np.float64] | None = None,
        normalisation_stats: dict[str, Any] | None = None,
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
        Evaluates the GP and the acquisition function at the given points, and negates
        the acquisition values (PSO minimises).

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
        lower_bound (list[float]): Lower bound of the optimization space.
        upper_bound (list[float]): Upper bound of the optimization space.
        n_particles (int | None): Number of particles, defaults to 200.
        options (dict[str, float] | None): hyperparameter values. Defaults to
            {"c1": 2.0, "c2": 2.0, "w": 0.9}
        seed (int | None): random seed.
    """

    def __init__(
        self,
        objective: PSOObjective,
        lower_bound: list[float],
        upper_bound: list[float],
        n_particles: int | None = None,
        options: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        xl = np.array(lower_bound, dtype=np.float64)
        xu = np.array(upper_bound, dtype=np.float64)
        bounds = (xl, xu)

        if n_particles is None:
            n_particles = 200
        if options is None:
            options = {"c1": 2.0, "c2": 2.0, "w": 0.9}

        self.objective = objective
        if seed is not None:
            np.random.seed(seed)
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
    ) -> tuple[None | NDArray[np.float64], None | NDArray[np.float64], bool]:
        """
        Runs the particle swarm optimisation. Stops early if no point in the initial
        swarm satisfies the thresholds.

        Returns:
            cost (NDArray[np.float64]): value of PSO at points.
            positions (NDArray[np.float64]): positions of particles.
            stop_optimisation (bool): whether to stop because no point is feasible.
        """

        # Evaluate the initial swarm: sets stop_optimisation if nothing is feasible.
        self.objective(self.optimizer.swarm.position)

        if self.objective.stop_optimisation:
            return None, None, True

        cost, best_pos = self.optimizer.optimize(
            self.objective, iters=100, verbose=False
        )

        return cost, best_pos, False
