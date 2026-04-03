import torch
import matplotlib.pyplot as plt


def contour_gp_plot(test_x, model, dataset):
    x_1 = torch.linspace(-5, 5, 100)
    x_2 = torch.linspace(-5, 5, 100)
    x1, x2 = torch.meshgrid(x_1, x_2, indexing="xy")

    x_grid = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=1).float()
    x_grid_std = (x_grid - dataset.x_mean) / dataset.x_std

    predictions = model.predict(x_grid_std)
    mean_std = predictions[:, 0]

    mean = mean_std * dataset.y_std + dataset.y_mean
    predictions_mean = mean.reshape(100, 100)

    test_x_orig = test_x * dataset.x_std + dataset.x_mean

    input_1 = test_x_orig[:, 0]
    input_2 = test_x_orig[:, 1]

    return input_1, input_2, x1, x2, predictions_mean


def plot_pred_vs_true(y_true, y_pred):
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.scatter(y_true, y_pred)
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs. True Values")
    plt.show()
