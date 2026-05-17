"""Phase 3B - Unified Predictors.

Wraps model inference securely. Provides safe decoding of integer outputs
into human-readable domain labels without requiring disk-based label encoders.
"""

from typing import Any, Dict, List
import numpy as np

from inference.loaders import registry_cache, TF_AVAILABLE
from inference.preprocessors import preprocess_for_task, preprocess_for_dnn

import logging
logger = logging.getLogger("inference.predictors")

# Hardcoded domain maps to guarantee exact alignment with Phase 2 training data
CROP_CLASSES = sorted([
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
    'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
    'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
    'coconut', 'cotton', 'jute', 'coffee'
])

FERTILITY_CLASSES = ["Low", "Medium", "High"]

DEFICIENCY_CLASSES = [
    "Balanced",
    "Nitrogen deficient",
    "Phosphorus deficient",
    "Potassium deficient"
]


def predict_crop_xgboost(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run Crop classification using the best XGBoost model."""
    info = registry_cache.get_best_model_info("crop")
    if not info:
        return {"error": "Crop model not found in registry."}
        
    features = info.get("features", [])
    model = registry_cache.load_model("crop")
    if model is None:
        return {"error": "Failed to load Crop model artifact."}

    # Preprocess
    try:
        X = preprocess_for_task(input_dict, "crop", features)
    except Exception as e:
        return {"error": str(e)}

    # Predict
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    
    return {
        "prediction": CROP_CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {CROP_CLASSES[i]: float(probs[i]) for i in range(len(probs))}
    }


def predict_fertility_xgboost(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run Fertility classification using the best XGBoost model."""
    info = registry_cache.get_best_model_info("fertility")
    if not info:
        return {"error": "Fertility model not found in registry."}
        
    features = info.get("features", [])
    model = registry_cache.load_model("fertility")
    if model is None:
        return {"error": "Failed to load Fertility model artifact."}

    try:
        X = preprocess_for_task(input_dict, "fertility", features)
    except Exception as e:
        return {"error": str(e)}

    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    
    return {
        "prediction": FERTILITY_CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {FERTILITY_CLASSES[i]: float(probs[i]) for i in range(len(probs))}
    }


def predict_deficiency_xgboost(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run Deficiency classification using the best XGBoost model."""
    info = registry_cache.get_best_model_info("deficiency")
    if not info:
        return {"error": "Deficiency model not found in registry."}
        
    features = info.get("features", [])
    model = registry_cache.load_model("deficiency")
    if model is None:
        return {"error": "Failed to load Deficiency model artifact."}

    try:
        X = preprocess_for_task(input_dict, "deficiency", features)
    except Exception as e:
        return {"error": str(e)}

    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    
    return {
        "prediction": DEFICIENCY_CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {DEFICIENCY_CLASSES[i]: float(probs[i]) for i in range(len(probs))}
    }


def predict_dnn_multitask(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run multi-task DNN to jointly predict crop, fertility, and deficiency."""
    if not TF_AVAILABLE:
        return {"error": "TensorFlow not installed. Cannot run DNN."}
        
    info = registry_cache.get_best_model_info("crop+fertility+deficiency")
    if not info:
        return {"error": "DNN model not found in registry."}
        
    features = info.get("features", [])
    model = registry_cache.load_model("crop+fertility+deficiency")
    if model is None:
        return {"error": "Failed to load DNN model artifact."}

    try:
        X = preprocess_for_dnn(input_dict, features)
    except Exception as e:
        return {"error": str(e)}

    # Predict
    # DNN returns a list of outputs: [crop_probs, fertility_probs, deficiency_probs]
    outputs = model.predict(X, verbose=0)
    crop_probs = outputs[0][0]
    fert_probs = outputs[1][0]
    def_probs = outputs[2][0]
    
    crop_idx = int(np.argmax(crop_probs))
    fert_idx = int(np.argmax(fert_probs))
    def_idx = int(np.argmax(def_probs))
    
    return {
        "crop": {
            "prediction": CROP_CLASSES[crop_idx],
            "confidence": float(crop_probs[crop_idx]),
        },
        "fertility": {
            "prediction": FERTILITY_CLASSES[fert_idx],
            "confidence": float(fert_probs[fert_idx]),
        },
        "deficiency": {
            "prediction": DEFICIENCY_CLASSES[def_idx],
            "confidence": float(def_probs[def_idx]),
        }
    }
