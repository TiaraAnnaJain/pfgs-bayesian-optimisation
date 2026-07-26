from pathlib import Path

from plotting.metrics_plots import plot_single_metric, plot_multiple_metrics
from tracking.csv_loader import from_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METHODS = {
    "PFGS": PROJECT_ROOT / "results/pfgs",
    "BO UCB": PROJECT_ROOT / "results/bo/UCB",
    "BO EI": PROJECT_ROOT / "results/bo/EI",
}

METRICS = [
    {
        "column": "train_data_max",
        "label": "Train data maximum",
        "fig_prefix": "train_data_max",
    },
    {
        "column": "entire_space_rrmse",
        "label": "Entire space RRMSE",
        "fig_prefix": "entire_rrmse",
    },
    {
        "column": "design_space_rrmse",
        "label": "Design space RRMSE",
        "fig_prefix": "design_rrmse",
    },
]


def main():
    data = {}

    for metric in METRICS:
        metric_data = {}

        for method, folder in METHODS.items():
            x, y = from_csv(
                folder=str(folder),
                x_column="iteration",
                y_column=metric["column"],
            )
            metric_data[method] = {
                "x": x,
                "y": y,
            }

        data[metric["column"]] = metric_data

    for metric in METRICS:
        metric_data = data[metric["column"]]
        for method, info in METHODS.items():
            plot_single_metric(
                x_data=metric_data[method]["x"],
                y_data=metric_data[method]["y"],
                y_label=metric["label"],
                fig_name=f"{metric['fig_prefix']}_{method.lower().replace(' ', '_')}",
                root=PROJECT_ROOT,
                title=f"{metric['label']} in {method}",
            )

        curves = {method: metric_data[method]["y"] for method in METHODS}
        longest_x = max(
            (metric_data[method]["x"] for method in METHODS),
            key=len,
        )
        plot_multiple_metrics(
            x_data=longest_x,
            curves=curves,
            y_label=metric["label"],
            fig_name=f"{metric['fig_prefix']}_comparison",
            root=PROJECT_ROOT,
            title=f"{metric['label']} comparison between methods",
        )

    for method, info in METHODS.items():
        curves = {
            "Entire space": data["entire_space_rrmse"][method]["y"],
            "Design space": data["design_space_rrmse"][method]["y"],
        }
        plot_multiple_metrics(
            x_data=data["entire_space_rrmse"][method]["x"],
            curves=curves,
            y_label="RRMSE",
            fig_name=f"rrmse_{method.lower().replace(' ', '_')}",
            root=PROJECT_ROOT,
            title=f"Entire space vs Design space RRMSE in {method}",
        )


if __name__ == "__main__":
    main()
