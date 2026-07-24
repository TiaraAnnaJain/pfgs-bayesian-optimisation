"""Contains function that calculates model metrics"""

from typing import Any, Dict

from data.data_models import Dataset
from gp_model.analysis import compute_rrmse


def calculate_model_metrics(
    model: Any, datasets: Dict[str, Dataset], normalisation_stats: Dict[str, Dict]
) -> Dict[str, float]:
    metrics = {}

    for name, dataset in datasets.items():
        recipe_dataset = dataset.copy()
        recipe_dataset.dataset_type = "recipe"
        recipe_dataset.y_data = None
        recipe_dataset.y_labels = None

        recipe_dataset.set_stats(normalisation_stats)
        recipe_dataset.normalise()

        predictions = model.predict(recipe_dataset)

        predictions.set_stats(normalisation_stats)
        predictions.de_normalise()

        metrics[f"{name} RRMSE"] = compute_rrmse(
            predictions.y_data.reshape(-1, 1), dataset.y_data
        )

    return metrics
