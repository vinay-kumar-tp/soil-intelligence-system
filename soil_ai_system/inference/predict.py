import numpy as np
import joblib
import tensorflow as tf
from config import SAVED_MODELS_PATH, FEATURE_COLS, CROP_LABELS
from inference.preprocess_input import preprocess_single_input
from inference.postprocess import format_output
from explainability.shap_engine import get_top_features, contrastive_explanation
from recommendation_engine.rules import full_recommendation
from utils.logger import get_logger

logger = get_logger("inference", "inference.log")

xgb_crop = joblib.load(f"{SAVED_MODELS_PATH}xgboost_crop.pkl")
xgb_fert = joblib.load(f"{SAVED_MODELS_PATH}xgboost_fertility.pkl")
xgb_def = joblib.load(f"{SAVED_MODELS_PATH}xgboost_deficiency.pkl")
dnn_model = tf.keras.models.load_model(f"{SAVED_MODELS_PATH}dnn_multitask.h5")
encoders = joblib.load(f"{SAVED_MODELS_PATH}label_encoders.pkl")


def run_full_inference(raw_input: dict) -> dict:
    """Run preprocessing, prediction, explainability, and recommendations.

    Args:
        raw_input (dict): Raw input payload from a user.

    Returns:
        dict: Formatted inference response with predictions and explanations.

    Side Effects:
        - Loads persisted models and encoders at module import.
    """
    logger.info(
        f"Inference request: N={raw_input.get('N')}, P={raw_input.get('P')}, K={raw_input.get('K')}"
    )

    X_scaled, soil_health_score = preprocess_single_input(raw_input)

    crop_proba = xgb_crop.predict_proba(X_scaled)[0]
    crop_idx = int(np.argmax(crop_proba))
    crop_label = encoders["crop"].inverse_transform([crop_idx])[0]
    crop_conf = float(crop_proba[crop_idx])

    fert_proba = xgb_fert.predict_proba(X_scaled)[0]
    fert_label = encoders["fertility_grade"].inverse_transform([int(np.argmax(fert_proba))])[0]
    fert_conf = float(np.max(fert_proba))

    def_proba = xgb_def.predict_proba(X_scaled)[0]
    def_label = encoders["nutrient_status"].inverse_transform([int(np.argmax(def_proba))])[0]

    contrastive = contrastive_explanation(
        dnn_model, X_scaled, crop_label, CROP_LABELS, [c for c in FEATURE_COLS], encoders
    )

    shap_top = get_top_features(
        xgb_crop.get_booster().get_score(importance_type="gain"), FEATURE_COLS
    )

    recs = full_recommendation(
        N=raw_input["N"],
        P=raw_input["P"],
        K=raw_input["K"],
        ph=raw_input["ph"],
        moisture=raw_input["moisture"],
        temperature=raw_input["temperature"],
        humidity=raw_input["humidity"],
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
