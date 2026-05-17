"""Phase 3D - Explainability Serving.

Generates local SHAP explanations for individual inference requests,
providing quantitative transparency into model decisions.
"""

import logging
from typing import Any, Dict, List

import numpy as np

from inference.loaders import registry_cache
from inference.predictors import CROP_CLASSES

logger = logging.getLogger("inference.explainers")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def generate_local_explanation(
    model: Any, 
    X_scaled: np.ndarray, 
    feature_names: List[str]
) -> Dict[str, Any]:
    """Generate SHAP-based feature importance for a single inference request.
    
    Args:
        model: Loaded XGBoost model.
        X_scaled: Scaled feature array of shape (1, N).
        feature_names: List of feature names matching X_scaled.
        
    Returns:
        Dict mapping feature names to their absolute SHAP contribution.
    """
    if not SHAP_AVAILABLE:
        return {"error": "SHAP library not installed"}
        
    try:
        # TreeExplainer is fast for single inference and requires no background data
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled, check_additivity=False)
        
        # Multiclass XGBoost returns a list of arrays (one per class)
        # We aggregate the mean absolute impact across all classes for simplicity
        if isinstance(shap_values, list):
            mean_impacts = np.mean([np.abs(sv[0]) for sv in shap_values], axis=0)
        else:
            mean_impacts = np.abs(shap_values[0])
            
        importance = {
            feat: float(val) 
            for feat, val in zip(feature_names, mean_impacts)
        }
        # Sort by importance descending
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return {"feature_contributions": sorted_importance}
        
    except Exception as e:
        logger.error("Failed to generate local explanation: %s", e)
        return {"error": "Explanation generation failed"}


def generate_contrastive_explanation(
    model: Any, 
    X_scaled: np.ndarray, 
    feature_names: List[str]
) -> Dict[str, Any]:
    """Generate contrastive explanation (Why Crop A and not Crop B?)."""
    try:
        probs = model.predict_proba(X_scaled)[0]
        top2_idx = np.argsort(probs)[::-1][:2]
        
        if len(top2_idx) < 2:
            return {"error": "Not enough classes for contrastive explanation"}
            
        top_idx, runner_up_idx = top2_idx[0], top2_idx[1]
        
        top_name = CROP_CLASSES[top_idx]
        runner_name = CROP_CLASSES[runner_up_idx]
        
        gap = float(probs[top_idx] - probs[runner_up_idx])
        
        # Simple rule-based contrastive string for API response
        explanation_str = (
            f"'{top_name.capitalize()}' is preferred over '{runner_name.capitalize()}' "
            f"because it fits the provided environmental metrics better by a confidence margin of {gap*100:.1f}%."
        )
        
        return {
            "predicted_crop": top_name,
            "runner_up_crop": runner_name,
            "confidence_gap": round(gap, 4),
            "explanation": explanation_str
        }
    except Exception as e:
        logger.error("Failed to generate contrastive explanation: %s", e)
        return {"error": "Contrastive explanation failed"}
