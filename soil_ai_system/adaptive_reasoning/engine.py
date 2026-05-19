"""Phase 7 - Connected Agricultural Adaptive Intelligence Engine.

Integrates Knowledge Graph edges, session memory drifts, and live environmental 
signals to optimize, adjust, and score agricultural crop strategies.
"""
from typing import Dict, Any, List
import logging

from knowledge_graph.graph import evaluate_graph_compatibility, get_crop_relationships
from session_memory.memory_manager import global_session_memory

logger = logging.getLogger("adaptive_reasoning.engine")

def calculate_soil_health_rating(telemetry: Dict[str, Any]) -> int:
    """Computes a baseline soil health metric from N, P, K ratios and pH."""
    n_val = float(telemetry.get("N", 50))
    p_val = float(telemetry.get("P", 40))
    k_val = float(telemetry.get("K", 40))
    ph_val = float(telemetry.get("ph", 6.5))
    
    # pH optimality check (6.0 - 7.5 is ideal for most crops)
    ph_score = 30
    if 6.0 <= ph_val <= 7.5:
        ph_score = 30
    elif 5.0 <= ph_val < 6.0 or 7.5 < ph_val <= 8.5:
        ph_score = 20
    else:
        ph_score = 10
        
    # NPK balance check
    npk_sum = n_val + p_val + k_val
    npk_score = 70
    if npk_sum < 60:  # depleted
        npk_score = 30
    elif npk_sum > 350:  # excess toxicity
        npk_score = 40
        
    return ph_score + npk_score

def compute_agronomic_intelligence_score(
    ml_confidence: float,
    env_compatibility: float,
    graph_offset: float,
    soil_health: float,
    risk_penalties: float
) -> int:
    """Calculates the premium Agronomic Intelligence Graph Score (0 - 100)."""
    weighted = (
        (0.35 * ml_confidence) +
        (0.30 * env_compatibility) +
        (0.20 * graph_offset) +
        (0.15 * soil_health)
    )
    score = weighted - risk_penalties
    return max(0, min(100, int(score)))

def execute_adaptive_reasoning(
    input_data: Dict[str, Any],
    base_predictions: Dict[str, Any],
    weather_intel: Dict[str, Any]
) -> Dict[str, Any]:
    """Ties together ML predictions, weather alerts, session drifts, and KG edges.
    
    Args:
        input_data: Manual telemetry input parameters.
        base_predictions: Predictor outputs (crop, fertility, deficiency).
        weather_intel: Live weather compatibility metrics.
        
    Returns:
        Structured Adaptive Intelligence payload.
    """
    # 1. Fetch Session Memory metrics & recurring patterns
    history = global_session_memory.get_history()
    recurring_gaps = global_session_memory.detect_recurring_deficiencies()
    drift = global_session_memory.detect_environmental_drift()
    
    # 2. Extract ML crop results
    crop_res = base_predictions.get("crop", {})
    predicted_crop = crop_res.get("prediction", "N/A").lower()
    ml_confidence = float(crop_res.get("confidence", 85.0))
    
    # 3. Fetch Knowledge Graph relationships for the crop
    kg_compat = evaluate_graph_compatibility(predicted_crop, input_data)
    
    # 4. Extract Weather environmental alignment score
    compatibility = weather_intel.get("compatibility", {})
    env_align = float(compatibility.get("alignment_score", 100.0))
    
    # Compute base risk penalties
    risks = compatibility.get("risks", {})
    drought = risks.get("drought_risk", "Low")
    flood = risks.get("flood_sensitivity", "Low")
    heat = risks.get("heat_stress", "Low")
    
    risk_penalties = 0.0
    if drought == "High": risk_penalties += 15.0
    elif drought == "Medium": risk_penalties += 5.0
    
    if flood == "High": risk_penalties += 10.0
    elif flood == "Medium": risk_penalties += 3.0
    
    if heat == "High": risk_penalties += 10.0
    
    # 5. Apply Session Adaptive Weighting Adjustments
    adaptive_weight_notices = []
    
    # Adjustment A: If recurring dry/drought drift is spotted in session, penalize water-thirsty crops
    kg_rules = get_crop_relationships(predicted_crop)
    if kg_rules and kg_rules.get("water_intensity") == "extreme":
        if drift.get("drought_trend") or drift.get("rainfall_trend") == "drying":
            risk_penalties += 20.0
            adaptive_weight_notices.append(
                f"Drying environmental drift detected in session memory. Water-intensive {predicted_crop.capitalize()} confidence penalized."
            )
            
    # Adjustment B: If Nitrogen/Potassium deficiencies recur continuously in session, boost organic soil recovery recommendations
    rebuild_urgency = "standard"
    if any(g in ["nitrogen", "potassium"] for g in [g.lower() for g in recurring_gaps]):
        rebuild_urgency = "CRITICAL RECOVERY"
        adaptive_weight_notices.append(
            f"Persistent recurring {', '.join(recurring_gaps)} deficiency detected. Soil rebuilding actions prioritized."
        )

    # 6. Compute Final Agronomic Intelligence Graph Score (0 - 100)
    soil_health = calculate_soil_health_rating(input_data)
    graph_base_score = 50 + kg_compat.get("score_offset", 0) + (kg_compat.get("sustainability_bonus", 0) * 4)
    graph_score = max(0, min(100, graph_base_score))
    
    intelligence_score = compute_agronomic_intelligence_score(
        ml_confidence=ml_confidence,
        env_compatibility=env_align,
        graph_offset=graph_score,
        soil_health=soil_health,
        risk_penalties=risk_penalties
    )
    
    # 7. Generate Premium Long-Form Agronomic Report Narratives
    narratives = []
    
    # Synthesis
    if env_align >= 80 and graph_score >= 70:
        narratives.append(
            f"The field demonstrates highly optimized conditions for {predicted_crop.capitalize()} cultivation. "
            f"Local telemetry is consistent with regional weather sensors, and the crop matches the natural {input_data.get('soil_texture', 'local')} soil graph preferences."
        )
    else:
        narratives.append(
            f"The field exhibits environmental constraints for {predicted_crop.capitalize()} cultivation. "
            f"Cross-referencing indicates key alignment offsets with real-time variables, advising targeted soil treatment before sowing."
        )
        
    # Recovery suggestions
    if rebuild_urgency == "CRITICAL RECOVERY":
        narratives.append(
            "CRITICAL SOIL RECOVERY PLAN: Prioritize rich organic mulching, deep green manure crop cover (e.g. Sesbania), and bio-fertilizer inoculations to counter systemic nutrient depletion."
        )
    elif input_data.get("N", 50) < 40:
        narratives.append(
            "SOIL HEALTH UPDATE: Mild nitrogen depletion detected. Incorporate organic leguminous crop cycles (such as groundnut) to naturally capture atmospheric nitrogen."
        )

    # Outlook
    irr_type = input_data.get("irrigation_type", "rain-fed")
    if risks.get("drought_risk") != "Low" and irr_type.lower() == "rain-fed":
        narratives.append(
            "OUTLOOK WARNING: Dry climate conditions forecast. Under rain-fed configurations, a transition to precision drip irrigation is strongly recommended to protect against heat-induced transpiration loss."
        )
    else:
        narratives.append(
            "OUTLOOK ADVISORY: Climate moisture balance is sufficient. Maintain standard canal/borewell schedules and monitor local soil drainage profiles."
        )
        
    # Sustainability
    kg_factors = kg_compat.get("sustainability_factors", [])
    if kg_factors:
        narratives.append(
            f"SUSTAINABILITY INSIGHTS: According to relationship graph nodes, {predicted_crop.capitalize()} cultivation carries a sustainability score of {kg_rules.get('sustainability_score', 80)}/100. "
            f"Key environmental aspects to monitor include: {', '.join(kg_factors)}."
        )

    return {
        "intelligence_score": intelligence_score,
        "soil_health_rating": soil_health,
        "kg_score": graph_score,
        "kg_match_details": {
            "matching": kg_compat.get("matching_factors", []),
            "conflicting": kg_compat.get("conflicting_factors", []),
            "sustainability_bonus": kg_compat.get("sustainability_bonus", 0)
        },
        "session_drift": drift,
        "recurring_deficiencies": recurring_gaps,
        "adaptive_weight_notices": adaptive_weight_notices,
        "soil_rebuild_urgency": rebuild_urgency,
        "long_form_report": narratives
    }
