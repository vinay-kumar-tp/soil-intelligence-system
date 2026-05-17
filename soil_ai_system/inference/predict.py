"""Full inference pipeline with lazy model loading.

Runs preprocessing, prediction, SHAP explainability, and recommendations
for a single input. Models are loaded on first call to avoid import-time
crashes when model files don't yet exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from config import CROP_LABELS, SAVED_MODELS_PATH, CROP_PROCESSED_FEATURE_COLS
from inference.preprocess_input import preprocess_single_input
from inference.postprocess import format_output
from recommendation_engine.rules import full_recommendation
from utils.logger import get_logger

logger = get_logger("inference", "inference.log")


def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    base_dir = Path(__file__).resolve().parents[1]
    return (base_dir / relative_path).resolve()


# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------

_models: dict = {}


def _load_models() -> dict:
    """Load all inference models lazily (on first call).

    Returns:
        dict: Loaded models and encoders.

    Raises:
        FileNotFoundError: When required model files are missing.
    """
    if _models:
        return _models

    import joblib

    models_path = _resolve_path(SAVED_MODELS_PATH)

    # XGBoost models
    xgb_crop_path = models_path / "xgboost_crop.pkl"
    xgb_fert_path = models_path / "xgboost_fertility.pkl"
    xgb_def_path = models_path / "xgboost_deficiency.pkl"

    missing = []
    for path in [xgb_crop_path, xgb_fert_path]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"Required model files not found: {missing}. "
            "Run training pipeline first (python -m training.train_all)."
        )

    _models["xgb_crop"] = joblib.load(str(xgb_crop_path))
    _models["xgb_fert"] = joblib.load(str(xgb_fert_path))
    _models["xgb_def"] = joblib.load(str(xgb_def_path)) if xgb_def_path.exists() else None

    # DNN model (optional — may not exist on CPU-only setups)
    dnn_path = models_path / "dnn_multitask.h5"
    if dnn_path.exists():
        import tensorflow as tf
        _models["dnn"] = tf.keras.models.load_model(str(dnn_path))
    else:
        _models["dnn"] = None
        logger.warning("DNN model not found at %s — SHAP contrastive disabled", dnn_path)

    # Label encoders
    encoder_path = models_path / "label_encoders.pkl"
    if encoder_path.exists():
        _models["encoders"] = joblib.load(str(encoder_path))
    else:
        # Try crop pipeline encoders
        from config import PIPELINE_ARTIFACTS, CROP_DATASET_KEY
        alt_path = _resolve_path(PIPELINE_ARTIFACTS[CROP_DATASET_KEY]) / "label_encoders.pkl"
        if alt_path.exists():
            payload = joblib.load(str(alt_path))
            _models["encoders"] = payload.get("encoders", payload)
        else:
            _models["encoders"] = {}
            logger.warning("No label encoders found")

    logger.info("Inference models loaded successfully")
    return _models


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_full_inference(raw_input: dict) -> dict:
    """Run preprocessing, prediction, explainability, and recommendations.

    Args:
        raw_input (dict): Raw input payload from a user.

    Returns:
        dict: Formatted inference response with predictions and explanations.
    """
    logger.info(
        "Inference request: N=%s, P=%s, K=%s",
        raw_input.get("N"), raw_input.get("P"), raw_input.get("K"),
    )

    models = _load_models()
    X_scaled, soil_health_score = preprocess_single_input(raw_input)

    # Crop prediction
    xgb_crop = models["xgb_crop"]
    crop_proba = xgb_crop.predict_proba(X_scaled)[0]
    crop_idx = int(np.argmax(crop_proba))
    crop_conf = float(crop_proba[crop_idx])

    encoders = models.get("encoders", {})

    if "crop" in encoders:
        crop_label = encoders["crop"].inverse_transform([crop_idx])[0]
    elif crop_idx < len(CROP_LABELS):
        crop_label = CROP_LABELS[crop_idx]
    else:
        crop_label = f"class_{crop_idx}"

    # Fertility prediction
    xgb_fert = models["xgb_fert"]
    fert_proba = xgb_fert.predict_proba(X_scaled)[0]
    fert_idx = int(np.argmax(fert_proba))
    fert_conf = float(np.max(fert_proba))

    if "fertility_grade" in encoders:
        fert_label = encoders["fertility_grade"].inverse_transform([fert_idx])[0]
    else:
        fert_labels = ["Low", "Medium", "High"]
        fert_label = fert_labels[fert_idx] if fert_idx < len(fert_labels) else f"grade_{fert_idx}"

    # Deficiency prediction
    xgb_def = models.get("xgb_def")
    if xgb_def is not None:
        def_proba = xgb_def.predict_proba(X_scaled)[0]
        def_idx = int(np.argmax(def_proba))
        if "nutrient_status" in encoders:
            def_label = encoders["nutrient_status"].inverse_transform([def_idx])[0]
        else:
            def_labels = ["Nitrogen deficient", "Phosphorus deficient", "Potassium deficient", "Balanced"]
            def_label = def_labels[def_idx] if def_idx < len(def_labels) else f"status_{def_idx}"
    else:
        def_label = "Unknown"

    # SHAP top features (feature importance fallback)
    feature_names = [c for c in CROP_PROCESSED_FEATURE_COLS if True]  # copy
    try:
        from explainability.shap_engine import get_top_features
        shap_top = get_top_features(
            xgb_crop.get_booster().get_score(importance_type="gain"),
            feature_names,
        )
    except Exception as exc:
        logger.warning("SHAP feature importance failed: %s", exc)
        shap_top = []

    # Contrastive explanation (requires DNN)
    dnn = models.get("dnn")
    contrastive = {}
    if dnn is not None:
        try:
            from explainability.shap_engine import contrastive_explanation
            contrastive = contrastive_explanation(
                dnn, X_scaled, crop_label, CROP_LABELS, feature_names, encoders,
            )
        except Exception as exc:
            logger.warning("Contrastive explanation failed: %s", exc)

    # Recommendations
    recs = full_recommendation(
        N=raw_input.get("N", 0),
        P=raw_input.get("P", 0),
        K=raw_input.get("K", 0),
        ph=raw_input.get("ph", 7.0),
        moisture=raw_input.get("moisture", 50),
        temperature=raw_input.get("temperature", 25),
        humidity=raw_input.get("humidity", 60),
        predicted_crop=crop_label,
        season=raw_input.get("season"),
        state=raw_input.get("state"),
    )

    return format_output(
        crop=crop_label,
        confidence_crop=crop_conf,
        fertility=fert_label,
        confidence_fert=fert_conf,
        deficiency=def_label,
        recs=recs,
        shap_top=shap_top,
        contrastive=contrastive,
        soil_health_score=soil_health_score,
    )
