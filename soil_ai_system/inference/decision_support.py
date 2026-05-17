"""Phase 6 - Agronomic Decision Support & Hybrid Intelligence Engine.

Provides deep agricultural reasoning, confidence tier mapping, multi-crop comparison, 
scientific soil health narratives, prioritized actions, and hybrid intelligence scoring.
"""

from typing import Any, Dict, List
import numpy as np

# Scientific crop requirements database
CROP_AGRONOMIC_PROFILES = {
    "rice": {
        "opt_rain": (1200, 3000), "opt_temp": (20, 38), "opt_hum": (70, 95), "opt_ph": (5.0, 7.0),
        "desc": "Water-intensive cereal crop requiring high moisture and swampy or clay loam conditions."
    },
    "maize": {
        "opt_rain": (500, 1500), "opt_temp": (18, 32), "opt_hum": (55, 80), "opt_ph": (5.5, 7.5),
        "desc": "Widely adaptable crop requiring warm temperatures and well-drained loamy soils."
    },
    "chickpea": {
        "opt_rain": (300, 700), "opt_temp": (15, 28), "opt_hum": (30, 60), "opt_ph": (6.0, 8.0),
        "desc": "Dry-climate legume crop highly susceptible to waterlogging but excellent for nitrogen fixation."
    },
    "kidneybeans": {
        "opt_rain": (400, 1000), "opt_temp": (15, 25), "opt_hum": (40, 70), "opt_ph": (5.5, 6.5),
        "desc": "Nutrient-rich legume crop preferring moderate rain and slightly acidic to neutral well-drained soil."
    },
    "pigeonpeas": {
        "opt_rain": (600, 1200), "opt_temp": (18, 35), "opt_hum": (45, 80), "opt_ph": (5.0, 7.5),
        "desc": "Drought-resistant pulse crop highly suitable for semi-arid tropics and intercropping."
    },
    "mothbeans": {
        "opt_rain": (200, 500), "opt_temp": (25, 40), "opt_hum": (30, 60), "opt_ph": (6.0, 7.5),
        "desc": "Extremely drought-hardy legume suitable for arid and sandy soils."
    },
    "mungbean": {
        "opt_rain": (400, 900), "opt_temp": (20, 35), "opt_hum": (40, 70), "opt_ph": (6.0, 7.5),
        "desc": "Short-duration pulse crop that performs well in warm, humid climates with moderate soils."
    },
    "blackgram": {
        "opt_rain": (600, 1000), "opt_temp": (25, 35), "opt_hum": (45, 80), "opt_ph": (6.0, 7.5),
        "desc": "Highly nutritious pulse crop requiring warm weather and well-drained clayey soils."
    },
    "lentil": {
        "opt_rain": (350, 700), "opt_temp": (10, 25), "opt_hum": (30, 60), "opt_ph": (6.0, 8.0),
        "desc": "Cool-season pulse crop highly sensitive to high humidity and water logging."
    },
    "pomegranate": {
        "opt_rain": (500, 1200), "opt_temp": (15, 35), "opt_hum": (30, 60), "opt_ph": (5.5, 7.5),
        "desc": "Resilient fruit crop suitable for semi-arid climates, preferring calcareous well-drained soils."
    },
    "banana": {
        "opt_rain": (1500, 3000), "opt_temp": (20, 35), "opt_hum": (65, 90), "opt_ph": (5.5, 8.0),
        "desc": "Tropical fruit crop requiring constant warm moisture, rich organic matter, and high potassium."
    },
    "mango": {
        "opt_rain": (750, 1500), "opt_temp": (20, 36), "opt_hum": (40, 75), "opt_ph": (5.5, 7.0),
        "desc": "Perennial deep-rooted fruit tree requiring dry dry-periods for fruit development."
    },
    "grapes": {
        "opt_rain": (400, 1000), "opt_temp": (15, 32), "opt_hum": (30, 65), "opt_ph": (6.0, 7.5),
        "desc": "Vine crop requiring good drainage, warm dry summers, and sensitive to excessive moisture."
    },
    "watermelon": {
        "opt_rain": (300, 800), "opt_temp": (22, 35), "opt_hum": (35, 65), "opt_ph": (5.5, 7.0),
        "desc": "Warm-season vine requiring highly sandy loam soils, low humidity, and high solar radiation."
    },
    "muskmelon": {
        "opt_rain": (300, 800), "opt_temp": (20, 35), "opt_hum": (35, 65), "opt_ph": (5.5, 7.0),
        "desc": "Slightly drought-hardy fruit crop requiring sandy soils, heat, and minimal humidity during harvest."
    },
    "apple": {
        "opt_rain": (600, 1500), "opt_temp": (5, 25), "opt_hum": (40, 70), "opt_ph": (5.5, 6.8),
        "desc": "Temperate fruit crop requiring chilling hours and moderately acidic, nutrient-dense soils."
    },
    "orange": {
        "opt_rain": (800, 1600), "opt_temp": (15, 32), "opt_hum": (45, 80), "opt_ph": (5.5, 7.5),
        "desc": "Citrus crop requiring uniform irrigation, good soil aeration, and highly vulnerable to frost."
    },
    "papaya": {
        "opt_rain": (1000, 2000), "opt_temp": (22, 35), "opt_hum": (60, 85), "opt_ph": (6.0, 7.0),
        "desc": "Herbaceous tropical plant requiring rich organic soils and extremely sensitive to water logging."
    },
    "coconut": {
        "opt_rain": (1000, 2500), "opt_temp": (20, 35), "opt_hum": (60, 90), "opt_ph": (5.0, 8.0),
        "desc": "Resilient coastal crop requiring steady warm sun, highly sand-tolerant, and thrives in high humidity."
    },
    "cotton": {
        "opt_rain": (500, 1200), "opt_temp": (22, 35), "opt_hum": (50, 80), "opt_ph": (5.5, 8.5),
        "desc": "Cash crop requiring dry sunny harvest periods and deep, moisture-retentive black cotton soils."
    },
    "jute": {
        "opt_rain": (1200, 2500), "opt_temp": (24, 38), "opt_hum": (70, 95), "opt_ph": (6.0, 7.5),
        "desc": "Fibre crop requiring hot, humid tropical climates and deep rich alluvial soils."
    },
    "coffee": {
        "opt_rain": (1000, 2200), "opt_temp": (15, 28), "opt_hum": (60, 85), "opt_ph": (5.0, 6.5),
        "desc": "Shade-loving plantation crop requiring cool humid highlands and slightly acidic, organic-rich soil."
    }
}

# Healthy thresholds for baseline comparisons
NUTRIENT_THRESHOLDS = {
    "N": {"low": 50, "high": 120},
    "P": {"low": 30, "high": 75},
    "K": {"low": 35, "high": 90},
    "ph": {"acidic": 5.8, "alkaline": 7.3}
}

def analyze_crop_suitability(crop: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates specific advantages, risks, and limiting factors for a crop based on telemetry."""
    profile = CROP_AGRONOMIC_PROFILES.get(crop.lower())
    if not profile:
        return {
            "advantages": ["Standard climatic alignment."],
            "risks": ["Nutrient status requires careful monitoring."],
            "limiting_factors": ["None observed."],
            "score_mod": 0
        }
        
    advantages = []
    risks = []
    limiting_factors = []
    score_mod = 0
    
    # 1. Rain check
    rain = inputs.get("rainfall", 0)
    opt_rain = profile["opt_rain"]
    if opt_rain[0] <= rain <= opt_rain[1]:
        advantages.append(f"Optimal rainfall ({rain:.1f} mm) matches water requirements.")
        score_mod += 10
    elif rain < opt_rain[0]:
        limiting_factors.append(f"Insufficient rainfall ({rain:.1f} mm vs ideal {opt_rain[0]} mm).")
        risks.append("Moisture stress risk requires crop irrigation support.")
        score_mod -= 15
    else:
        risks.append("High rainfall risk could lead to waterlogging or root damage.")
        score_mod -= 10

    # 2. Temperature check
    temp = inputs.get("temperature", 25)
    opt_temp = profile["opt_temp"]
    if opt_temp[0] <= temp <= opt_temp[1]:
        advantages.append(f"Ideal climate temperature ({temp:.1f}°C) supports growth phases.")
        score_mod += 5
    else:
        limiting_factors.append(f"Temperature exposure ({temp:.1f}°C) is outside optimal {opt_temp[0]}-{opt_temp[1]}°C.")
        risks.append("Thermal stress may affect yield or flowering rates.")
        score_mod -= 10

    # 3. Humidity check
    hum = inputs.get("humidity", 50)
    opt_hum = profile["opt_hum"]
    if opt_hum[0] <= hum <= opt_hum[1]:
        advantages.append(f"Optimal atmospheric humidity ({hum:.1f}%) regulates transpiration.")
        score_mod += 5
    else:
        risks.append(f"Suboptimal humidity ({hum:.1f}%) might promote fungal diseases or high transpiration.")
        score_mod -= 5

    # 4. pH check
    ph = inputs.get("ph", 6.5)
    opt_ph = profile["opt_ph"]
    if opt_ph[0] <= ph <= opt_ph[1]:
        advantages.append(f"Soil acidity level (pH {ph:.1f}) enables optimal nutrient absorption.")
        score_mod += 10
    else:
        limiting_factors.append(f"Soil pH {ph:.1f} is outside crop's ideal range ({opt_ph[0]}-{opt_ph[1]}).")
        risks.append("Altered pH reduces bioavailability of critical soil micronutrients.")
        score_mod -= 15

    # 5. Core nutrient deficiencies
    n = inputs.get("N", 0)
    p = inputs.get("P", 0)
    k = inputs.get("K", 0)
    
    if n < NUTRIENT_THRESHOLDS["N"]["low"]:
        risks.append("Low nitrogen will restrict vegetative growth and foliage development.")
        score_mod -= 5
    if p < NUTRIENT_THRESHOLDS["P"]["low"]:
        risks.append("Low phosphorus blocks robust root establishment.")
        score_mod -= 5
    if k < NUTRIENT_THRESHOLDS["K"]["low"]:
        risks.append("Low potassium reduces natural drought and pest resistance.")
        score_mod -= 5
        
    if not advantages:
        advantages.append("Soil holds baseline compatibility.")
    if not risks:
        risks.append("No critical agronomic risks predicted.")
    if not limiting_factors:
        limiting_factors.append("No critical climatic limits observed.")

    return {
        "advantages": advantages[:3],
        "risks": risks[:3],
        "limiting_factors": limiting_factors[:2],
        "score_mod": score_mod
    }

def generate_decision_support(
    inputs: Dict[str, Any],
    crop_res: Dict[str, Any],
    fertility_res: Dict[str, Any],
    deficiency_res: Dict[str, Any]
) -> Dict[str, Any]:
    """Orchestrates Phase 6 Agronomic Reasoning and Decision Intelligence on top of standard predictions."""
    
    # --- 6A: Confidence-Aware Intelligence ---
    base_confidence = crop_res.get("confidence", 0.5)
    if base_confidence >= 0.75:
        conf_tier = "HIGH CONFIDENCE"
        conf_message = f"{crop_res['prediction'].capitalize()} strongly matches the current environmental and soil conditions."
    elif base_confidence >= 0.40:
        conf_tier = "MODERATE CONFIDENCE"
        conf_message = f"The field partially supports {crop_res['prediction']} cultivation, but some conditions may reduce yield stability."
    else:
        conf_tier = "LOW CONFIDENCE"
        conf_message = f"Current field conditions are weakly aligned with {crop_res['prediction']} cultivation. Exercise caution."

    # --- 6B: Top-K Comparative Crop Reasoning ---
    all_probs = crop_res.get("all_probabilities", {})
    top_3_crops = []
    if all_probs:
        # Sort crop probabilities descending
        sorted_crops = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for name, prob in sorted_crops:
            suit = analyze_crop_suitability(name, inputs)
            # Suitability score is based on model probability scaled to 100, combined with environmental checks
            suitability_score = max(5, min(99, int(prob * 100 + suit["score_mod"])))
            
            top_3_crops.append({
                "crop": name.capitalize(),
                "probability": float(prob),
                "suitability_score": suitability_score,
                "description": CROP_AGRONOMIC_PROFILES.get(name, {}).get("desc", "Agronomic details pending local classification."),
                "advantages": suit["advantages"],
                "risks": suit["risks"],
                "limiting_factors": suit["limiting_factors"]
            })
    else:
        # Fallback if probability array is missing
        top_3_crops.append({
            "crop": crop_res.get("prediction", "Unknown").capitalize(),
            "probability": base_confidence,
            "suitability_score": int(base_confidence * 100),
            "description": "Primary recommended crop.",
            "advantages": ["Standard climate alignment."],
            "risks": ["Standard monitoring recommended."],
            "limiting_factors": ["None."]
        })

    # --- 6C: Agronomic Narrative Engine ---
    primary_crop = crop_res.get("prediction", "Unknown").lower()
    ph = inputs.get("ph", 6.5)
    n = inputs.get("N", 0)
    p = inputs.get("P", 0)
    k = inputs.get("K", 0)
    rain = inputs.get("rainfall", 0)
    humidity = inputs.get("humidity", 50)
    
    narratives = []
    
    # pH Interpretation
    if ph < NUTRIENT_THRESHOLDS["ph"]["acidic"]:
        narratives.append(f"Soil acidity is elevated (pH {ph:.1f}), which restricts optimal uptake of vital macronutrients like nitrogen and phosphorus.")
    elif ph > NUTRIENT_THRESHOLDS["ph"]["alkaline"]:
        narratives.append(f"Soil is moderately alkaline (pH {ph:.1f}), potentially binding iron and manganese, rendering them unavailable for crop root systems.")
    else:
        narratives.append(f"Soil pH is balanced in the neutral optimal range (pH {ph:.1f}), facilitating efficient root bioavailability and nutrient transport.")

    # Climate Compatibility
    if primary_crop in CROP_AGRONOMIC_PROFILES:
        req = CROP_AGRONOMIC_PROFILES[primary_crop]
        if req["opt_rain"][0] <= rain <= req["opt_rain"][1]:
            narratives.append(f"Atmospheric inputs are exceptional: Rainfall of {rain:.1f} mm meets water-budget targets for {primary_crop.capitalize()}.")
        else:
            narratives.append(f"Rainfall levels ({rain:.1f} mm) diverge from ideal targets ({req['opt_rain'][0]}-{req['opt_rain'][1]} mm) for water-sensitive {primary_crop.capitalize()}.")
            
        if req["opt_hum"][0] <= humidity <= req["opt_hum"][1]:
            narratives.append(f"Relative humidity ({humidity:.1f}%) matches {primary_crop.capitalize()}'s optimal transpiration requirements.")
    else:
        narratives.append("Atmospheric and temperature factors are compatible with general agricultural practices.")

    # Nutrient Balance
    low_nutrients = []
    if n < NUTRIENT_THRESHOLDS["N"]["low"]: low_nutrients.append("Nitrogen")
    if p < NUTRIENT_THRESHOLDS["P"]["low"]: low_nutrients.append("Phosphorus")
    if k < NUTRIENT_THRESHOLDS["K"]["low"]: low_nutrients.append("Potassium")
    
    if low_nutrients:
        narratives.append(f"Chemical analysis highlights suboptimal levels of {', '.join(low_nutrients)}, requiring strategic fertilizer top-offs to avoid nutrient deficiency stunted growth.")
    else:
        narratives.append("Macro-nutrient concentrations (N-P-K) are rich, providing excellent metabolic fuel for high-yield cycles.")

    narrative_full = " ".join(narratives)

    # --- 6D: Recommendation Prioritization ---
    high_priority = []
    moderate_priority = []
    optional_priority = []
    
    # High Priority triggers (Critical health issues)
    if ph < 5.5:
        high_priority.append("Apply agricultural lime (calcium carbonate) to buffer highly acidic soil (neutralize toxic aluminum/iron).")
    elif ph > 8.0:
        high_priority.append("Incorporate elemental sulfur or gypsum to reduce soil alkalinity and unlock bound micro-elements.")
        
    if fertility_res.get("prediction") == "Low":
        high_priority.append("Incorporate rich organic humus, compost, or animal manure to rebuild soil organic carbon (SOC) levels.")

    deficiency = deficiency_res.get("prediction", "")
    if "Nitrogen" in deficiency:
        high_priority.append("Apply fast-release Nitrogen fertilizer (e.g., Urea or Ammonium Sulfate) to correct severe foliage chlorosis.")
    elif "Phosphorus" in deficiency:
        high_priority.append("Supplement soil with water-soluble Phosphate (e.g., Diammonium Phosphate - DAP or Triple Super Phosphate).")
    elif "Potassium" in deficiency:
        high_priority.append("Distribute Muriate of Potash (MOP) or Potassium Sulfate to buffer drought and disease resistance.")

    # Moderate Priority triggers (Safety & maintenance)
    if not high_priority:
        # Default safety action if soil is completely balanced
        moderate_priority.append("Execute standard N-P-K maintenance routine. Monitor micro-nutrient ratios.")
    else:
        if fertility_res.get("prediction") == "Medium":
            moderate_priority.append("Integrate dynamic green manure cover crops (e.g., alfalfa) to naturally elevate nitrogen levels.")
        moderate_priority.append("Introduce systematic soil aeration to promote friendly aerobic microbial colonies.")

    # Optional / Optimizations (Long-term gains)
    optional_priority.append("Implement localized drip irrigation to regulate root zone hydration and prevent fertilizer washing/runoff.")
    if primary_crop in ["rice", "jute", "cotton"]:
        optional_priority.append(f"Consider rotational pulse farming (e.g., chickpea/lentils) next cycle to naturally restore nitrogen reserves.")
    else:
        optional_priority.append("Introduce cover cropping to protect soil top layers from summer wind erosion.")

    # Guarantee at least one instruction per list
    if not high_priority:
        high_priority.append("Continue routine weekly soil telemetry scans. No critical soil correctors required.")
    if not moderate_priority:
        moderate_priority.append("Conduct a professional deep-core soil diagnostic before the next sowing season.")

    # --- 6E: Research-Grade Hybrid Intelligence Scoring ---
    # Baseline score derived from ML crop confidence + fertility balance + pH correctness
    score = int(base_confidence * 40)  # Up to 40 points from model prediction confidence
    
    # Up to 30 points from fertility status
    fert = fertility_res.get("prediction", "Medium")
    if fert == "High": score += 30
    elif fert == "Medium": score += 20
    else: score += 10
    
    # Up to 30 points from pH matching
    ph_dist = abs(ph - 6.5)
    if ph_dist <= 0.5: score += 30
    elif ph_dist <= 1.2: score += 20
    else: score += 10
    
    # Cap between 10 and 98 to keep it scientifically realistic (100% perfect soils are a myth)
    hybrid_score = max(10, min(98, score))
    
    score_reasons = []
    if base_confidence >= 0.75:
        score_reasons.append("High neural model prediction agreement elevated baseline certainty.")
    else:
        score_reasons.append("Slight model variance in secondary crop candidates reduced confidence.")
        
    if ph_dist <= 0.5:
        score_reasons.append("Perfect soil pH balance highly optimized nutrient solubility.")
    else:
        score_reasons.append("Suboptimal soil pH slightly lowered general nutrient absorption efficiency.")
        
    if deficiency == "Balanced" and fert != "Low":
        score_reasons.append("Absence of severe macro-nutrient deficiencies guaranteed high yield stability.")
    else:
        score_reasons.append("Predicted macronutrient deficiency posed minor yield stability hazards.")

    return {
        "confidence": {
            "tier": conf_tier,
            "message": conf_message
        },
        "top_k_crops": top_3_crops,
        "narrative": narrative_full,
        "prioritized_actions": {
            "high": high_priority,
            "moderate": moderate_priority,
            "optional": optional_priority
        },
        "hybrid_intelligence_score": {
            "score": hybrid_score,
            "reasons": score_reasons[:2]
        }
    }
