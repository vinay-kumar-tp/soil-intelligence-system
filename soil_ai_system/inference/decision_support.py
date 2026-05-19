"""Phase 6 - Agronomic Decision Support & Hybrid Intelligence Engine.

Provides deep agricultural reasoning, confidence tier mapping, multi-crop comparison, 
scientific soil health narratives, prioritized actions, and hybrid intelligence scoring.
"""

from typing import Any, Dict, List
import numpy as np

# Scientific crop requirements database
# Scientific crop requirements database with seasonal and lifecycle parameters
CROP_AGRONOMIC_PROFILES = {
    "rice": {
        "opt_rain": (1200, 3000), "opt_temp": (20, 38), "opt_hum": (70, 95), "opt_ph": (5.0, 7.0),
        "seasons": ["kharif"], "duration": "seasonal",
        "desc": "Water-intensive cereal crop requiring high moisture and swampy or clay loam conditions."
    },
    "maize": {
        "opt_rain": (500, 1500), "opt_temp": (18, 32), "opt_hum": (55, 80), "opt_ph": (5.5, 7.5),
        "seasons": ["kharif", "rabi"], "duration": "seasonal",
        "desc": "Widely adaptable crop requiring warm temperatures and well-drained loamy soils."
    },
    "chickpea": {
        "opt_rain": (300, 700), "opt_temp": (15, 28), "opt_hum": (30, 60), "opt_ph": (6.0, 8.0),
        "seasons": ["rabi"], "duration": "seasonal",
        "desc": "Dry-climate legume crop highly susceptible to waterlogging but excellent for nitrogen fixation."
    },
    "kidneybeans": {
        "opt_rain": (400, 1000), "opt_temp": (15, 25), "opt_hum": (40, 70), "opt_ph": (5.5, 6.5),
        "seasons": ["rabi"], "duration": "seasonal",
        "desc": "Nutrient-rich legume crop preferring moderate rain and slightly acidic to neutral well-drained soil."
    },
    "pigeonpeas": {
        "opt_rain": (600, 1200), "opt_temp": (18, 35), "opt_hum": (45, 80), "opt_ph": (5.0, 7.5),
        "seasons": ["kharif"], "duration": "seasonal",
        "desc": "Drought-resistant pulse crop highly suitable for semi-arid tropics and intercropping."
    },
    "mothbeans": {
        "opt_rain": (200, 500), "opt_temp": (25, 40), "opt_hum": (30, 60), "opt_ph": (6.0, 7.5),
        "seasons": ["kharif"], "duration": "seasonal",
        "desc": "Extremely drought-hardy legume suitable for arid and sandy soils."
    },
    "mungbean": {
        "opt_rain": (400, 900), "opt_temp": (20, 35), "opt_hum": (40, 70), "opt_ph": (6.0, 7.5),
        "seasons": ["summer", "kharif"], "duration": "seasonal",
        "desc": "Short-duration pulse crop that performs well in warm, humid climates with moderate soils."
    },
    "blackgram": {
        "opt_rain": (600, 1000), "opt_temp": (25, 35), "opt_hum": (45, 80), "opt_ph": (6.0, 7.5),
        "seasons": ["kharif", "rabi"], "duration": "seasonal",
        "desc": "Highly nutritious pulse crop requiring warm weather and well-drained clayey soils."
    },
    "lentil": {
        "opt_rain": (350, 700), "opt_temp": (10, 25), "opt_hum": (30, 60), "opt_ph": (6.0, 8.0),
        "seasons": ["rabi"], "duration": "seasonal",
        "desc": "Cool-season pulse crop highly sensitive to high humidity and water logging."
    },
    "pomegranate": {
        "opt_rain": (500, 1200), "opt_temp": (15, 35), "opt_hum": (30, 60), "opt_ph": (5.5, 7.5),
        "seasons": ["perennial"], "duration": "perennial",
        "desc": "Resilient fruit crop suitable for semi-arid climates, preferring calcareous well-drained soils."
    },
    "banana": {
        "opt_rain": (1500, 3000), "opt_temp": (20, 35), "opt_hum": (65, 90), "opt_ph": (5.5, 8.0),
        "seasons": ["perennial"], "duration": "perennial",
        "desc": "Tropical fruit crop requiring constant warm moisture, rich organic matter, and high potassium."
    },
    "mango": {
        "opt_rain": (750, 1500), "opt_temp": (20, 36), "opt_hum": (40, 75), "opt_ph": (5.5, 7.0),
        "seasons": ["perennial"], "duration": "perennial",
        "desc": "Perennial deep-rooted fruit tree requiring dry periods for fruit development."
    },
    "grapes": {
        "opt_rain": (400, 1000), "opt_temp": (15, 32), "opt_hum": (30, 65), "opt_ph": (6.0, 7.5),
        "seasons": ["perennial"], "duration": "perennial",
        "desc": "Vine crop requiring good drainage, warm dry summers, and sensitive to excessive moisture."
    },
    "watermelon": {
        "opt_rain": (300, 800), "opt_temp": (22, 35), "opt_hum": (35, 65), "opt_ph": (5.5, 7.0),
        "seasons": ["summer"], "duration": "seasonal",
        "desc": "Warm-season vine requiring highly sandy loam soils, low humidity, and high solar radiation."
    },
    "muskmelon": {
        "opt_rain": (300, 800), "opt_temp": (20, 35), "opt_hum": (35, 65), "opt_ph": (5.5, 7.0),
        "seasons": ["summer"], "duration": "seasonal",
        "desc": "Slightly drought-hardy fruit crop requiring sandy soils, heat, and minimal humidity during harvest."
    },
    "apple": {
        "opt_rain": (600, 1500), "opt_temp": (5, 25), "opt_hum": (40, 70), "opt_ph": (5.5, 6.8),
        "seasons": ["perennial"], "duration": "perennial",
        "desc": "Temperate fruit crop requiring chilling hours and moderately acidic, nutrient-dense soils."
    },
    "orange": {
        "opt_rain": (800, 1600), "opt_temp": (15, 32), "opt_hum": (45, 80), "opt_ph": (5.5, 7.5),
        "seasons": ["perennial"], "duration": "perennial",
        "desc": "Citrus crop requiring uniform irrigation, good soil aeration, and highly vulnerable to frost."
    },
    "papaya": {
        "opt_rain": (1000, 2000), "opt_temp": (22, 35), "opt_hum": (60, 85), "opt_ph": (6.0, 7.0),
        "seasons": ["perennial"], "duration": "perennial",
        "desc": "Herbaceous tropical plant requiring rich organic soils and extremely sensitive to water logging."
    },
    "coconut": {
        "opt_rain": (1000, 2500), "opt_temp": (20, 35), "opt_hum": (60, 90), "opt_ph": (5.0, 8.0),
        "seasons": ["perennial"], "duration": "perennial",
        "desc": "Resilient coastal crop requiring steady warm sun, highly sand-tolerant, and thrives in high humidity."
    },
    "cotton": {
        "opt_rain": (500, 1200), "opt_temp": (22, 35), "opt_hum": (50, 80), "opt_ph": (5.5, 8.5),
        "seasons": ["kharif"], "duration": "seasonal",
        "desc": "Cash crop requiring dry sunny harvest periods and deep, moisture-retentive black cotton soils."
    },
    "jute": {
        "opt_rain": (1200, 2500), "opt_temp": (24, 38), "opt_hum": (70, 95), "opt_ph": (6.0, 7.5),
        "seasons": ["kharif"], "duration": "seasonal",
        "desc": "Fibre crop requiring hot, humid tropical climates and deep rich alluvial soils."
    },
    "coffee": {
        "opt_rain": (1000, 2200), "opt_temp": (15, 28), "opt_hum": (60, 85), "opt_ph": (5.0, 6.5),
        "seasons": ["perennial"], "duration": "perennial",
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
        score_mod -= 20
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
        score_mod -= 15

    # 3. Humidity check
    hum = inputs.get("humidity", 50)
    opt_hum = profile["opt_hum"]
    if opt_hum[0] <= hum <= opt_hum[1]:
        advantages.append(f"Optimal atmospheric humidity ({hum:.1f}%) regulates transpiration.")
        score_mod += 5
    else:
        risks.append(f"Suboptimal humidity ({hum:.1f}%) might promote fungal diseases.")
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
        score_mod -= 20

    # 5. Core nutrient deficiencies
    n = inputs.get("N", 0)
    p = inputs.get("P", 0)
    k = inputs.get("K", 0)
    
    if n < NUTRIENT_THRESHOLDS["N"]["low"]:
        risks.append("Low nitrogen will restrict vegetative growth.")
        score_mod -= 5
    if p < NUTRIENT_THRESHOLDS["P"]["low"]:
        risks.append("Low phosphorus blocks robust root establishment.")
        score_mod -= 5
    if k < NUTRIENT_THRESHOLDS["K"]["low"]:
        risks.append("Low potassium reduces natural drought resistance.")
        score_mod -= 5
        
    # 6. Season and Duration Check (PHYSICS OVERRULE)
    season = inputs.get("seasonal_context", "").lower()
    crop_seasons = profile.get("seasons", [])
    duration = profile.get("duration", "seasonal")
    
    if season and crop_seasons:
        if season in crop_seasons:
            advantages.append(f"Aligned with seasonal context ({season.capitalize()} crop).")
            score_mod += 15
        elif "perennial" in crop_seasons:
            # Perennial crops are long-term, so we check if there's high stress or rain-fed limits
            if rain < 500 and inputs.get("irrigation_type", "").lower() == "rain-fed":
                limiting_factors.append(f"Perennial {crop.capitalize()} requires long-term water, but rain-fed rain is insufficient.")
                risks.append("Severe long-term soil moisture depletion risk.")
                score_mod -= 25
            else:
                advantages.append("Perennial crop with stable year-round growth profile.")
                score_mod += 5
        else:
            limiting_factors.append(f"Mismatched season (sowing {crop.capitalize()} in {season.capitalize()} is non-traditional).")
            risks.append(f"Sowing {crop.capitalize()} in {season.capitalize()} leads to cycle mismatch.")
            score_mod -= 30
            
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
    
    # Calculate suitability for all known crops to find both best and worst
    all_probs = crop_res.get("all_probabilities", {})
    all_crop_suitability = []
    
    for name in CROP_AGRONOMIC_PROFILES.keys():
        prob = all_probs.get(name, 0.0)
        suit = analyze_crop_suitability(name, inputs)
        
        # Physics-based suitability score calculation:
        # Starts at 70%, adjusted by score_mod (pH, Rain, Temp, Season, Soil type)
        phys_score = max(5, min(99, int(70 + suit["score_mod"])))
        
        # Truly Hybrid Suitability Score:
        if prob > 0.05:
            # Merge ML probability (up to 40%) with environment physics (up to 60%)
            suitability_score = max(5, min(99, int(prob * 40 + phys_score * 0.60)))
        else:
            # Allows highly suitable seasonal alternatives to bubble up even if the ML model predicted 0%
            suitability_score = max(5, min(95, int(phys_score * 0.85)))
            
        all_crop_suitability.append({
            "crop": name.capitalize(),
            "suitability_score": suitability_score,
            "description": CROP_AGRONOMIC_PROFILES[name]["desc"],
            "advantages": suit["advantages"],
            "risks": suit["risks"],
            "limiting_factors": suit["limiting_factors"],
            "raw_suit": suit
        })
        
    # Find suitability score of primary predicted crop
    primary_crop = crop_res.get("prediction", "Unknown").lower()
    primary_suitability_score = 50
    for item in all_crop_suitability:
        if item["crop"].lower() == primary_crop:
            primary_suitability_score = item["suitability_score"]
            break
            
    # --- 6A: Confidence-Aware Intelligence (Physics Overrule Check) ---
    base_confidence = crop_res.get("confidence", 0.5)
    if primary_suitability_score < 40:
        conf_tier = "CRITICAL MISMATCH"
        conf_message = f"WARNING: The ML model suggested {crop_res['prediction'].capitalize()}, but environmental physics show a CRITICAL MISMATCH (Suitability: {primary_suitability_score}%). We highly recommend sowing seasonal winter/Rabi crops instead."
    elif base_confidence >= 0.75 and primary_suitability_score >= 70:
        conf_tier = "HIGH CONFIDENCE"
        conf_message = f"{crop_res['prediction'].capitalize()} strongly matches both neural predictions and seasonal environment physics."
    elif base_confidence >= 0.40 and primary_suitability_score >= 50:
        conf_tier = "MODERATE CONFIDENCE"
        conf_message = f"The field partially supports {crop_res['prediction']} cultivation, but seasonal factors suggest some yield volatility."
    else:
        conf_tier = "LOW CONFIDENCE"
        conf_message = f"Current field conditions are weakly aligned with {crop_res['prediction']} cultivation. Exercise caution."

    # --- 6B: Top-K Comparative Crop Reasoning ---
    # Top 3 most suitable crops (sorted descending by suitability score)
    sorted_suitable = sorted(all_crop_suitability, key=lambda x: x["suitability_score"], reverse=True)
    top_3_crops = sorted_suitable[:3]
    
    # Crops to Avoid (sorted ascending by suitability score, excluding the top 3)
    avoid_candidates = sorted(all_crop_suitability, key=lambda x: x["suitability_score"])
    crops_to_avoid = []
    for item in avoid_candidates:
        if item["crop"] not in [c["crop"] for c in top_3_crops]:
            # Generate a scientific reason why it must be avoided
            limits = item["limiting_factors"]
            risks = item["risks"]
            reason = "Mismatched soil/climate parameters."
            if limits and limits[0] != "No critical climatic limits observed.":
                reason = f"{limits[0]} {risks[0] if risks else ''}"
            elif risks and risks[0] != "No critical agronomic risks predicted.":
                reason = risks[0]
                
            crops_to_avoid.append({
                "crop": item["crop"],
                "suitability_score": item["suitability_score"],
                "reason": reason
            })
            if len(crops_to_avoid) >= 3: # get worst 3
                break

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
        "crops_to_avoid": crops_to_avoid,
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
