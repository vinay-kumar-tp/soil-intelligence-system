"""Phase 3C - Recommendation Intelligence Layer.

Translates ML predictions into deterministic, rule-guided, human-readable
agronomic recommendations.
"""

from typing import Any, Dict

# Basic agronomic rules for demonstration of the intelligence layer
FERTILIZER_SUGGESTIONS = {
    "Nitrogen deficient": "Apply Urea (46-0-0) or Ammonium Nitrate.",
    "Phosphorus deficient": "Apply Diammonium Phosphate (DAP) or Single Super Phosphate (SSP).",
    "Potassium deficient": "Apply Muriate of Potash (MOP) or Potassium Sulfate.",
    "Balanced": "Maintain current fertilizer schedule; soil nutrients are balanced."
}

def generate_recommendations(
    raw_input: Dict[str, Any],
    crop_prediction: str,
    fertility_prediction: str,
    deficiency_prediction: str
) -> Dict[str, Any]:
    """Generate structured, explainable recommendations.
    
    Args:
        raw_input: The raw dictionary of user inputs (N, P, K, humidity, etc.)
        crop_prediction: The crop name predicted (e.g., 'rice')
        fertility_prediction: The fertility grade (e.g., 'Low', 'High')
        deficiency_prediction: The nutrient status (e.g., 'Nitrogen deficient')
        
    Returns:
        Dict containing structured recommendations.
    """
    recs = {
        "crop_rationale": [],
        "soil_health_actions": [],
        "fertilizer_recommendation": ""
    }
    
    # 1. Crop rationale based on input values (rule-based explainability layer)
    crop = crop_prediction.lower()
    if crop == "rice":
        if raw_input.get("humidity", 0) > 75:
            recs["crop_rationale"].append("High humidity strongly favors rice.")
        if raw_input.get("rainfall", 0) > 150:
            recs["crop_rationale"].append("Heavy rainfall aligns with rice water requirements.")
    elif crop == "wheat":
        if raw_input.get("temperature", 30) < 25:
            recs["crop_rationale"].append("Cooler temperatures are optimal for wheat.")
        if raw_input.get("rainfall", 100) < 100:
            recs["crop_rationale"].append("Lower rainfall matches wheat drought tolerance.")
    else:
        recs["crop_rationale"].append(f"Soil metrics and climate are highly suitable for {crop_prediction}.")
        
    # 2. Fertility Actions
    if fertility_prediction == "Low":
        recs["soil_health_actions"].append("Soil organic carbon or macronutrients are critically low. Consider organic compost integration.")
    elif fertility_prediction == "High":
        recs["soil_health_actions"].append("Soil is highly fertile. Avoid over-fertilization to prevent nutrient runoff.")

    if raw_input.get("ph", 7) < 5.5:
        recs["soil_health_actions"].append("Soil is highly acidic. Consider applying agricultural lime.")
    elif raw_input.get("ph", 7) > 7.5:
        recs["soil_health_actions"].append("Soil is highly alkaline. Consider applying elemental sulfur.")

    # 3. Fertilizer Suggestion
    recs["fertilizer_recommendation"] = FERTILIZER_SUGGESTIONS.get(
        deficiency_prediction, 
        "Consult local agronomic guidelines."
    )
    
    return recs
