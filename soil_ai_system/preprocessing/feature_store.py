"""Save and load complete feature pipeline artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib

from config import (
    KMEANS_FILENAME,
    LABEL_ENCODERS_FILENAME,
    MODEL_VERSION,
    PIPELINE_ARTIFACTS,
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
    base_path = Path(SAVED_MODELS_PATH)
    if base_path.name == version_name:
        return base_path.resolve()
    return (base_path / version_name).resolve()


def _resolve_artifact_dir(
    artifact_dir: Optional[str],
    dataset_key: Optional[str],
    version: Optional[str],
) -> Path:
    """Resolve the artifact directory for a dataset pipeline or versioned path."""
    if artifact_dir:
        return Path(artifact_dir).resolve()
    if dataset_key and dataset_key in PIPELINE_ARTIFACTS:
        return Path(PIPELINE_ARTIFACTS[dataset_key]).resolve()
    return _get_version_path(version)


def save_pipeline(
    scaler,
    encoders,
    kmeans,
    version: Optional[str] = None,
    artifact_dir: Optional[str] = None,
    dataset_key: Optional[str] = None,
) -> None:
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
    path = _resolve_artifact_dir(artifact_dir, dataset_key, version)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path / SCALER_FILENAME)
    joblib.dump(encoders, path / LABEL_ENCODERS_FILENAME)
    joblib.dump(kmeans, path / KMEANS_FILENAME)
    LOGGER.info("Pipeline saved to %s", path)


def load_pipeline(
    version: Optional[str] = None,
    artifact_dir: Optional[str] = None,
    dataset_key: Optional[str] = None,
) -> Dict[str, object]:
    """Load preprocessing artifacts for a model version.

    Missing artifacts are returned as ``None`` rather than raising errors,
    since not every pipeline produces all three artifact types.

    Args:
        version (str | None): Model version folder name.
        artifact_dir (str | None): Explicit artifact directory override.
        dataset_key (str | None): Pipeline key for config-driven lookup.

    Returns:
        dict: Loaded scaler, encoders, and KMeans artifacts (or None).
    """
    path = _resolve_artifact_dir(artifact_dir, dataset_key, version)
    payload: Dict[str, object] = {"scaler": None, "encoders": {}, "kmeans": None}

    scaler_path = path / SCALER_FILENAME
    if scaler_path.exists():
        payload["scaler"] = joblib.load(scaler_path)
    else:
        LOGGER.warning("Scaler not found at %s", scaler_path)

    encoder_path = path / LABEL_ENCODERS_FILENAME
    if encoder_path.exists():
        loaded = joblib.load(encoder_path)
        payload["encoders"] = loaded.get("encoders", loaded) if isinstance(loaded, dict) else loaded
    else:
        LOGGER.warning("Label encoders not found at %s", encoder_path)

    kmeans_path = path / KMEANS_FILENAME
    if kmeans_path.exists():
        payload["kmeans"] = joblib.load(kmeans_path)

    LOGGER.info("Pipeline loaded from %s", path)
    return payload

