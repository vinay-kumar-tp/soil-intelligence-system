def format_output(
    crop,
    confidence_crop,
    fertility,
    confidence_fert,
    deficiency,
    recs,
    shap_top,
    contrastive,
    soil_health_score,
):
    """Format prediction outputs into a response payload.

    Args:
        crop (str): Predicted crop label.
        confidence_crop (float): Crop confidence score.
        fertility (str): Predicted fertility grade.
        confidence_fert (float): Fertility confidence score.
        deficiency (str): Predicted nutrient status.
        recs (dict): Recommendation outputs.
        shap_top (list[dict]): Top SHAP features.
        contrastive (dict): Contrastive explanation details.
        soil_health_score (float): Soil health score value.

    Returns:
        dict: Normalized API response payload.
    """
    return {
        "crop": crop,
        "confidence_crop": round(confidence_crop, 4),
        "fertility_grade": fertility,
        "confidence_fertility": round(confidence_fert, 4),
        "nutrient_status": deficiency,
        "fertilizer_recommendations": recs["fertilizer"],
        "irrigation_suggestion": recs["irrigation"],
        "seasonal_advice": recs["seasonal"],
        "crop_action_guide": recs["crop_guide"],
        "shap_top_features": shap_top,
        "contrastive_explanation": contrastive,
        "soil_health_score": round(soil_health_score, 1),
    }
