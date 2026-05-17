"""Model loading and prediction smoke tests.

These tests require trained model artifacts and are skipped when the
files don't exist (i.e. before Phase 2 training has been run).
"""

from pathlib import Path

import numpy as np
import pytest
import joblib
from config import SAVED_MODELS_PATH


def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    base_dir = Path(__file__).resolve().parents[1]
    return (base_dir / relative_path).resolve()


def _model_path(filename: str) -> Path:
    """Return the resolved path for a model file."""
    return _resolve_path(SAVED_MODELS_PATH + filename)


@pytest.mark.skipif(
    not Path(__file__).resolve().parents[1].joinpath(SAVED_MODELS_PATH, "xgboost_crop.pkl").exists(),
    reason="Model not trained yet",
)
def test_xgboost_loads():
    """Ensure the XGBoost crop model can be loaded.

    Args:
        None

    Returns:
        None
    """
    model = joblib.load(str(_model_path("xgboost_crop.pkl")))
    assert model is not None


@pytest.mark.skipif(
    not Path(__file__).resolve().parents[1].joinpath(SAVED_MODELS_PATH, "xgboost_crop.pkl").exists(),
    reason="Model not trained yet",
)
def test_xgboost_predicts():
    """Ensure the XGBoost model can run a prediction.

    Args:
        None

    Returns:
        None
    """
    model = joblib.load(str(_model_path("xgboost_crop.pkl")))
    # Use the crop feature count (8 processed features)
    from config import CROP_PROCESSED_FEATURE_COLS
    n_features = len(CROP_PROCESSED_FEATURE_COLS)
    dummy = np.random.rand(1, n_features)
    pred = model.predict(dummy)
    assert len(pred) == 1


@pytest.mark.skipif(
    not Path(__file__).resolve().parents[1].joinpath(SAVED_MODELS_PATH, "dnn_multitask.h5").exists(),
    reason="DNN model not trained yet",
)
def test_dnn_loads():
    """Ensure the DNN model can be loaded.

    Args:
        None

    Returns:
        None
    """
    import tensorflow as tf

    model = tf.keras.models.load_model(str(_model_path("dnn_multitask.h5")))
    assert model is not None
