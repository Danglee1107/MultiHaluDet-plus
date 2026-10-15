"""Append the final metrics of each pipeline run to one CSV file."""
import csv
import os
from datetime import datetime

RESULTS_CSV = "results/results.csv"
# Scalar keys of evaluate_all and compute_uncertainty_metrics, in column order.
METRICS = ("auc", "ap", "f1", "acc", "bal_acc", "mcc", "kappa", "sensitivity", "specificity",
           "brier_score", "log_loss", "ece")


def append_result(run: dict[str, object], metrics: dict[str, float], path: str = RESULTS_CSV) -> None:
    """Add one row: the time, the run settings, then the metrics with 4 decimals.

    A run that is repeated adds a new row; the time column tells the rows apart.
    """
    row = {"time": datetime.now().isoformat(timespec="seconds"), **run,
           **{name: f"{metrics[name]:.4f}" for name in METRICS}}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
