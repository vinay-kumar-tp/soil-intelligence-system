import numpy as np
import joblib
from config import SAVED_MODELS_PATH


def test_xgboost_loads():
    """Ensure the XGBoost crop model can be loaded.

    Args:
        None

    Returns:
        None
    """
    model = joblib.load(f"{SAVED_MODELS_PATH}xgboost_crop.pkl")
    assert model is not None


def test_xgboost_predicts():
    """Ensure the XGBoost model can run a prediction.

    Args:
        None

    Returns:
        None
    """
    model = joblib.load(f"{SAVED_MODELS_PATH}xgboost_crop.pkl")
    dummy = np.random.rand(1, 17)
    pred = model.predict(dummy)
    assert len(pred) == 1


def test_dnn_loads():
    """Ensure the DNN model can be loaded.

    Args:
        None

    Returns:
        None
    """
    import tensorflow as tf

    model = tf.keras.models.load_model(f"{SAVED_MODELS_PATH}dnn_multitask.h5")
    assert model is not None
