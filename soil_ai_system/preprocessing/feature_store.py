"""Save and load complete feature pipeline artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib

from config import (
    KMEANS_FILENAME,
    LABEL_ENCODERS_FILENAME,
    MODEL_VERSION,
    PREPROCESSING_LOG_FILE,
    SAVED_MODELS_PATH,
    SCALER_FILENAME,
)
from utils.logger import get_logger


LOGGER = get_logger("preprocessing.feature_store", PREPROCESSING_LOG_FILE)


ARTIFACTS = [SCALER_FILENAME, LABEL_ENCODERS_FILENAME, KMEANS_FILENAME]


def _get_version_path(version: Optional[str]) -> Path:
    """Build a versioned path for saved artifacts."""
    version_name = version or MODEL_VERSION
    return (Path(SAVED_MODELS_PATH) / version_name).resolve()


def save_pipeline(scaler, encoders, kmeans, version: Optional[str] = None) -> None:
    """Persist preprocessing artifacts for a model version.

    Args:
        scaler (object): Fitted scaler instance.
        encoders (dict): Label encoders mapping.
        kmeans (object): Fitted KMeans instance.
        version (str | None): Model version folder name.

    Returns:
        None

    Side Effects:
        - Writes artifacts to the saved_models directory.
    """
    path = _get_version_path(version)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path / SCALER_FILENAME)
    joblib.dump(encoders, path / LABEL_ENCODERS_FILENAME)
    joblib.dump(kmeans, path / KMEANS_FILENAME)
    LOGGER.info("Pipeline saved to %s", path)


def load_pipeline(version: Optional[str] = None) -> Dict[str, object]:
    """Load preprocessing artifacts for a model version.

    Args:
        version (str | None): Model version folder name.

    Returns:
        dict: Loaded scaler, encoders, and KMeans artifacts.
    """
    path = _get_version_path(version)
    payload = {
        "scaler": joblib.load(path / SCALER_FILENAME),
        "encoders": joblib.load(path / LABEL_ENCODERS_FILENAME),
        "kmeans": joblib.load(path / KMEANS_FILENAME),
    }
    LOGGER.info("Pipeline loaded from %s", path)
    return payload
