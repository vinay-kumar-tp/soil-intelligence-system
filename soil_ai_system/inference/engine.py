"""Phase 3B - Unified Prediction Engine.

Orchestrates the entire inference pipeline, tying together loaders,
preprocessors, predictors, explainers, and recommenders into a single
safe, robust, and unified JSON response.
"""

import logging
import time
from typing import Any, Dict

from inference.predictors import (
    predict_crop_xgboost,
    predict_deficiency_xgboost,
    predict_dnn_multitask,
    predict_fertility_xgboost,
)
from inference.explainers import generate_local_explanation, generate_contrastive_explanation
from inference.recommenders import generate_recommendations
from inference.loaders import registry_cache
from inference.preprocessors import preprocess_for_task
from inference.decision_support import generate_decision_support
from recommendation_engine.spatial_reasoning_engine import apply_spatial_reasoning
from weather.weather_context_engine import get_live_weather_context
from adaptive_reasoning.engine import execute_adaptive_reasoning
from session_memory.memory_manager import global_session_memory

logger = logging.getLogger("inference.engine")


def run_full_inference(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full AI orchestration pipeline.
    
    Args:
        input_data: Validated dictionary matching the API schema.
        
    Returns:
        Structured JSON dictionary containing predictions, explanations, and recommendations.
    """
    import copy
    input_data = copy.deepcopy(input_data)
    start_time = time.time()
    logger.info("Starting unified inference request.")
    
    # 1. Base Predictions via XGBoost pipelines
    crop_res = predict_crop_xgboost(input_data)
    fert_res = predict_fertility_xgboost(input_data)
    def_res = predict_deficiency_xgboost(input_data)
    
    # 2. Extract final labels for downstream rules
    # Fallback safely if prediction failed
    c_pred = crop_res.get("prediction", "Unknown")
    f_pred = fert_res.get("prediction", "Unknown")
    d_pred = def_res.get("prediction", "Unknown")
    
    # 3. Recommendations (Phase 3C)
    recs = generate_recommendations(input_data, c_pred, f_pred, d_pred)
    
    # 4. Phase 6 - Decision Support & Agronomic Intelligence (Hybrid Reasoning)
    decision_support = {}
    try:
        if "error" not in crop_res and "error" not in fert_res and "error" not in def_res:
            decision_support = generate_decision_support(input_data, crop_res, fert_res, def_res)
            # Apply Spatial Intelligence (Phase 6X)
            decision_support = apply_spatial_reasoning(input_data, decision_support)
    except Exception as e:
        logger.error("Decision Support Engine failed: %s", e)
        decision_support = {"error": f"Decision support failure: {str(e)}"}
    
    # 5. Explainability (Phase 3D)
    explanations = {}
    try:
        # Load best crop model and features for local SHAP
        crop_info = registry_cache.get_best_model_info("crop")
        if crop_info and "error" not in crop_res:
            features = crop_info.get("features", [])
            model = registry_cache.load_model("crop")
            if model:
                # Preprocess specifically for the explainer
                X_scaled = preprocess_for_task(input_data, "crop", features)
                
                # Generate explanations
                local_shap = generate_local_explanation(model, X_scaled, features)
                contrastive = generate_contrastive_explanation(model, X_scaled, features)
                
                explanations["feature_importance"] = local_shap
                explanations["contrastive"] = contrastive
    except Exception as e:
        logger.warning("Explainability generation failed: %s", e)
        explanations["error"] = "Failed to generate explanations."

    # 5B. Weather & Environmental Intelligence Orchestration (Phase 6Z)
    weather_intel = {}
    try:
        lat = input_data.get("latitude")
        lon = input_data.get("longitude")
        if lat is not None and lon is not None:
            weather_intel = get_live_weather_context(float(lat), float(lon), input_data)
        elif input_data.get("weather_context"):
            weather_intel = input_data.get("weather_context")
    except Exception as e:
        logger.warning("Weather Intelligence orchestration failed: %s", e)
        weather_intel = {"status": "error", "message": str(e)}

    # 5C. Adaptive & Connected Intelligence Layer (Phase 7)
    adaptive_intel = {}
    try:
        base_predictions = {
            "predictions": {
                "crop": crop_res,
                "fertility": fert_res,
                "deficiency": def_res,
            }
        }
        adaptive_intel = execute_adaptive_reasoning(input_data, base_predictions, weather_intel)
        # Store in session memory
        global_session_memory.add_entry(input_data, base_predictions)
    except Exception as e:
        logger.warning("Adaptive reasoning layer failed: %s", e)
        adaptive_intel = {"status": "error", "message": str(e)}

    # 6. Build Unified JSON Response
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    response = {
        "status": "success",
        "predictions": {
            "crop": crop_res,
            "fertility": fert_res,
            "deficiency": def_res,
        },
        "recommendations": recs,
        "decision_support": decision_support,
        "explanations": explanations,
        "weather_intelligence": weather_intel,
        "adaptive_intelligence": adaptive_intel,
        "metadata": {
            "inference_latency_ms": latency_ms,
            "model_versions": {
                "crop": "xgboost_v1",
                "fertility": "xgboost_v1",
                "deficiency": "xgboost_v1"
            }
        }
    }
    
    logger.info("Unified inference completed in %.2f ms", latency_ms)
    return response
