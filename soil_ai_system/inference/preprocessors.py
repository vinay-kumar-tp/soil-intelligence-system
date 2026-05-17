"""Phase 3B - Inference Preprocessing Pipeline.

Routes the incoming request payload to the correct feature engineering,
scaling, and formatting logic depending on the target task.
Ensures zero preprocessing leakage during inference.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from inference.loaders import registry_cache

logger = logging.getLogger("inference.preprocessors")


def _engineer_features(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply deterministic rules to generate derived features.
    
    Args:
        input_dict: Raw input dictionary from API.
        
    Returns:
        Dictionary augmented with engineered features.
    """
    data = input_dict.copy()
    
    # Generate fertility score (N+P+K)/3
    if all(k in data for k in ["N", "P", "K"]):
        data["fertility_score"] = (data["N"] + data["P"] + data["K"]) / 3.0
        
    # Generate soil quality index if all nutrients/context available
    req_sqi = ["N", "P", "K", "organic_carbon", "ph"]
    if all(k in data for k in req_sqi):
        data["soil_quality_index"] = (
            0.3 * data["N"]
            + 0.25 * data["P"]
            + 0.25 * data["K"]
            + 0.1 * data["organic_carbon"]
            - 0.1 * abs(data["ph"] - 6.5)
        )
        
    return data


def preprocess_for_task(input_dict: Dict[str, Any], task: str, expected_features: List[str]) -> np.ndarray:
    """Prepare the raw dictionary for the specific task model.
    
    Args:
        input_dict: Raw input dictionary from API.
        task: Target pipeline ('crop', 'fertility', 'deficiency', 'ensemble', etc.)
        expected_features: List of feature names expected by the model.
        
    Returns:
        np.ndarray: Scaled and ordered feature matrix (shape 1 x N).
    """
    # 1. Engineer features
    data = _engineer_features(input_dict)
    
    # 2. Extract strictly the required features in order, with safe imputation
    try:
        raw_list = []
        for f in expected_features:
            if f not in data:
                logger.warning("Imputing missing feature %s with 0.0 for task %s", f, task)
                data[f] = 0.0
            raw_list.append(data[f])
        raw_array = np.array([raw_list], dtype=np.float64)
    except Exception as e:
        logger.error("Failed to construct feature array for %s task: %s", task, e)
        raise ValueError(f"Feature construction failed: {e}")

    # 3. Apply appropriate scaler
    scaler_name = ""
    if "crop" in task or "deficiency" in task:
        # Deficiency relies on the crop processed dataset features
        scaler_name = "crop_pipeline/scaler.pkl"
    elif "fertility" in task:
        scaler_name = "fertility_pipeline/scaler.pkl"
        
    scaler = registry_cache.load_scaler(scaler_name)
    if scaler is None:
        logger.error("Failed to load scaler %s for task %s", scaler_name, task)
        # Fallback to unscaled if absolutely necessary (but usually fatal)
        return raw_array
        
    # 4. Scale
    scaled_array = scaler.transform(raw_array)
    return scaled_array


def preprocess_for_dnn(input_dict: Dict[str, Any], expected_features: List[str]) -> np.ndarray:
    """Preprocess data for the multi-task DNN.
    
    The DNN was trained on crop_processed.csv feature schemas.
    """
    return preprocess_for_task(input_dict, "crop", expected_features)
