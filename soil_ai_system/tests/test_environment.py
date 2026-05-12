"""Phase 0 smoke tests for environment and structure."""

from pathlib import Path


def _project_root() -> Path:
    """Return the project root path.

    Args:
        None

    Returns:
        pathlib.Path: Root directory for the project.
    """
    return Path(__file__).resolve().parents[1]


def test_project_structure_exists():
    """Ensure critical folders exist before moving to Phase 1.

    Args:
        None

    Returns:
        None
    """
    root = _project_root()
    required = [
        "datasets/raw",
        "datasets/processed",
        "notebooks",
        "preprocessing",
        "models",
        "ensemble",
        "training",
        "inference",
        "explainability",
        "recommendation_engine",
        "experiment_tracking",
        "api",
        "dashboard",
        "tests",
        "saved_models/v1",
        "logs",
        "reports",
    ]
    missing = [p for p in required if not (root / p).exists()]
    assert not missing, f"Missing required paths: {missing}"


def test_core_imports():
    """Verify key library imports succeed.

    Args:
        None

    Returns:
        None
    """
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
    import tensorflow  # noqa: F401
    import xgboost  # noqa: F401
    import shap  # noqa: F401
    import fastapi  # noqa: F401
    import pydantic  # noqa: F401


def test_tensorflow_gpu_check():
    """Confirm TensorFlow can query GPU devices (CPU-only is valid).

    Args:
        None

    Returns:
        None
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    assert isinstance(gpus, list)


def test_logger_writes_file():
    """Ensure logger creates the training log file.

    Args:
        None

    Returns:
        None

    Side Effects:
        - Writes a log entry to logs/training.log.
    """
    from utils.logger import get_logger

    root = _project_root()
    log_path = root / "logs" / "training.log"
    logger = get_logger("env_test", "training.log")
    logger.info("Smoke test log entry")
    assert log_path.exists()
