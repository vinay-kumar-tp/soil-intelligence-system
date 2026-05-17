import csv
import os
from datetime import datetime

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

CSV_LOG_PATH = "reports/experiment_log.csv"


def log_experiment(model_name, params, metrics, notes=""):
    """Log experiment parameters and metrics to MLflow and CSV.

    Args:
        model_name (str): Run or model identifier.
        params (dict): Parameter dictionary.
        metrics (dict): Metric dictionary.
        notes (str): Optional notes for the run.

    Returns:
        None

    Side Effects:
        - Writes experiment data to MLflow and reports/experiment_log.csv.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("notes", notes)
            mlflow.set_tag("timestamp", timestamp)

    row = {"timestamp": timestamp, "model": model_name, "notes": notes, **params, **metrics}
    file_exists = os.path.exists(CSV_LOG_PATH)
    with open(CSV_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    first_metric_val = next(iter(metrics.values()), 0)
    first_metric_key = next(iter(metrics.keys()), "metric")
    print(f"[ExperimentLog] {model_name} - {first_metric_key}={float(first_metric_val):.4f}")
