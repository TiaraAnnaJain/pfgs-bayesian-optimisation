"""Generates contour plots of trained GP model."""

from typing import Any, Optional, List, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from src.data.data_models import Dataset


def contour_slice_plot(
    trained_model: Any,
    train_dataset: Dataset,
    x_plot_labels: List[str],
    y_plot_label: str,
    fixed_x_values: Optional[Dict[str, float]] = None,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Generate a contour plot of the trained model for any two inputs specified.

    Parameters:
        trained_model (Any): The trained model to plot.
        train_dataset (Dataset): The training dataset.
        x_plot_labels (List[str]): Labels of input values to plot.
        y_plot_label (str): Label of output to plot.
        fixed_x_values (Optional[Dict[str, float]], optional): Fixed values of other
            inputs which are not to be plotted.

    Returns:
        training_sample_1 (NDArray[np.float64]): Training data values for input 1.
        training_sample_2 (NDArray[np.float64]): Training data values for input 2.
        x1 (NDArray[np.float64]): Meshgrid array containing the first input variable
            values used for the contour plot.
        x2 (NDArray[np.float64]): Meshgrid array containing the second input variable
            values used for the contour plot.
        preds (NDArray[np.float64]): Predicted mean output from the trained model over
            the x1-x2 grid.

    """
    if len(x_plot_labels) != 2:
        raise ValueError("Exactly two x labels must be provided.")

    for label in x_plot_labels:
        if label not in train_dataset.x_labels:
            raise ValueError(f"{label} is not in x_labels.")

    if y_plot_label not in train_dataset.y_labels:
        raise ValueError(f"{y_plot_label} is not in y_labels.")

    x1_index = train_dataset.x_labels.index(x_plot_labels[0])
    x2_index = train_dataset.x_labels.index(x_plot_labels[1])

    x1_lb = train_dataset.x_data[:, x1_index].min()
    x2_lb = train_dataset.x_data[:, x2_index].min()

    x1_ub = train_dataset.x_data[:, x1_index].max()
    x2_ub = train_dataset.x_data[:, x2_index].max()

    x_1 = np.linspace(x1_lb, x1_ub, 100)
    x_2 = np.linspace(x2_lb, x2_ub, 100)
    x1, x2 = np.meshgrid(x_1, x_2, indexing="xy")

    n_points = x1.size
    n_x = len(train_dataset.x_labels)

    x_grid = np.zeros((n_points, n_x))

    for i, label in enumerate(train_dataset.x_labels):
        if label == x_plot_labels[0]:
            x_grid[:, i] = x1.reshape(-1)

        elif label == x_plot_labels[1]:
            x_grid[:, i] = x2.reshape(-1)

        else:
            if fixed_x_values is not None and label in fixed_x_values:
                x_grid[:, i] = fixed_x_values[label]
            else:
                x_grid[:, i] = train_dataset.x_data[:, i].mean()

    x_grid_standardised = Dataset(
        dataset_type="recipe",
        x_data=x_grid,
        x_labels=train_dataset.x_labels,
    )

    normalisation_stats = train_dataset.get_stats()
    x_grid_standardised.set_stats(normalisation_stats)
    x_grid_standardised.normalise()

    predict_dataset = x_grid_standardised.copy()

    predictions = trained_model.predict(predict_dataset)

    predictions.set_stats(normalisation_stats)
    predictions.de_normalise()

    y_index = train_dataset.y_labels.index(y_plot_label)
    preds = predictions.y_data[:, y_index].reshape(100, 100)

    training_sample_1 = train_dataset.x_data[:, x1_index]
    training_sample_2 = train_dataset.x_data[:, x2_index]

    return training_sample_1, training_sample_2, x1, x2, preds
