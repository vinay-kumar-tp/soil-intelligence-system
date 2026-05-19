"""Phase 7A — Agronomic Knowledge Graph.

Defines the core agronomic schema, crop nodes, environmental relationships, 
traversal utilities, and recommendation enrichment hooks.
"""
from typing import Dict, Any, List, Set, Optional

# Define static node relationships
AGRONOMIC_KNOWLEDGE_GRAPH = {
    "rice": {
        "requires_humidity": "high",
        "prefers_soil": ["clay", "loamy", "black soil"],
        "sensitive_to": ["drought", "water_stress"],
        "demands": {"N": "high", "P": "medium", "K": "medium"},
        "irrigation_preference": ["canal irrigation", "borewell"],
        "agro_climatic_zones": ["Coastal", "Delta"],
        "sustainability_score": 75,
        "sustainability_factors": ["High water footprint", "Methane emissions risk if flooded continuously"],
        "nitrogen_fixer": False,
        "water_intensity": "extreme"
    },
    "maize": {
        "requires_humidity": "medium",
        "prefers_soil": ["loamy", "black soil", "red soil"],
        "sensitive_to": ["frost", "waterlogging"],
        "demands": {"N": "high", "P": "medium", "K": "high"},
        "irrigation_preference": ["borewell", "drip irrigation", "canal irrigation"],
        "agro_climatic_zones": ["Semi-Arid", "Dry Plains"],
        "sustainability_score": 82,
        "sustainability_factors": ["High nutrient demand", "Supports crop rotation nicely"],
        "nitrogen_fixer": False
    },
    "cotton": {
        "requires_humidity": "medium",
        "prefers_soil": ["black soil", "clay"],
        "sensitive_to": ["frost", "heavy_rain", "flood_sensitivity"],
        "demands": {"N": "medium", "P": "medium", "K": "high"},
        "irrigation_preference": ["borewell", "drip irrigation"],
        "agro_climatic_zones": ["Semi-Arid", "Dry Plains"],
        "sustainability_score": 78,
        "sustainability_factors": ["Deep taproot improves soil structure", "Highly pesticide dependent unless organic"],
        "nitrogen_fixer": False
    },
    "groundnut": {
        "requires_humidity": "medium",
        "prefers_soil": ["sandy", "red soil", "loamy"],
        "sensitive_to": ["drought", "frost"],
        "demands": {"N": "low", "P": "high", "K": "medium"},
        "irrigation_preference": ["rain-fed", "drip irrigation"],
        "agro_climatic_zones": ["Semi-Arid", "Dry Plains", "Coastal"],
        "sustainability_score": 96,
        "sustainability_factors": ["Leguminous nitrogen-fixer", "Improves organic carbon, highly sustainable"],
        "nitrogen_fixer": True
    },
    "sugarcane": {
        "requires_humidity": "high",
        "prefers_soil": ["clay", "black soil", "loamy"],
        "sensitive_to": ["frost", "drought"],
        "demands": {"N": "high", "P": "high", "K": "high"},
        "irrigation_preference": ["canal irrigation", "borewell"],
        "agro_climatic_zones": ["Coastal", "Delta", "Western Ghats"],
        "sustainability_score": 65,
        "sustainability_factors": ["Extremely high water and nutrient consumer", "Long duration exhausts soil profile"],
        "nitrogen_fixer": False,
        "water_intensity": "extreme"
    },
    "banana": {
        "requires_humidity": "high",
        "prefers_soil": ["clay", "loamy", "black soil"],
        "sensitive_to": ["drought", "high_winds"],
        "demands": {"N": "high", "P": "medium", "K": "high"},
        "irrigation_preference": ["borewell", "drip irrigation"],
        "agro_climatic_zones": ["Coastal", "Delta", "Western Ghats"],
        "sustainability_score": 74,
        "sustainability_factors": ["High potassium demands", "Requires structural windbreaks"],
        "nitrogen_fixer": False
    },
    "coconut": {
        "requires_humidity": "high",
        "prefers_soil": ["sandy", "loamy", "red soil"],
        "sensitive_to": ["drought", "frost"],
        "demands": {"N": "medium", "P": "low", "K": "high"},
        "irrigation_preference": ["drip irrigation", "borewell"],
        "agro_climatic_zones": ["Coastal", "Western Ghats"],
        "sustainability_score": 90,
        "sustainability_factors": ["Perennial deep root holds coastal sand dunes", "Highly resilient once established"],
        "nitrogen_fixer": False
    }
}

# Soil Nutrient Threshold reference (for graph matching)
NUTRIENT_LEVELS = {
    "N": {"low": 50, "high": 120},
    "P": {"low": 30, "high": 80},
    "K": {"low": 35, "high": 90}
}

def get_crop_relationships(crop_name: str) -> Optional[Dict[str, Any]]:
    """Traverses knowledge graph node for a specific crop."""
    return AGRONOMIC_KNOWLEDGE_GRAPH.get(crop_name.lower())

def evaluate_graph_compatibility(
    crop_name: str,
    telemetry: Dict[str, Any]
) -> Dict[str, Any]:
    """Computes relationship compatibility score and matching details.
    
    Returns:
        Dictionary containing match status, reinforcing factors, and penalties.
    """
    crop = crop_name.lower()
    rules = AGRONOMIC_KNOWLEDGE_GRAPH.get(crop)
    if not rules:
        return {
            "score_offset": 0,
            "matching_factors": [],
            "conflicting_factors": [],
            "sustainability_bonus": 0
        }
        
    matching = []
    conflicting = []
    score_offset = 0
    
    # 1. Soil Texture Preferences
    soil_text = str(telemetry.get("soil_texture", "")).lower()
    if soil_text:
        preferred_soils = [s.lower() for s in rules["prefers_soil"]]
        matched_soil = False
        for pref in preferred_soils:
            if pref in soil_text or soil_text in pref:
                matched_soil = True
                break
        if matched_soil:
            matching.append(f"Prefers {soil_text} texture matches graph relationship.")
            score_offset += 15
        else:
            conflicting.append(f"Soil texture ({soil_text}) differs from preferred ({', '.join(rules['prefers_soil'])}).")
            score_offset -= 10
            
    # 2. Agro-Climatic Zone Matches
    zone = str(telemetry.get("agro_climatic_zone", "")).lower()
    if zone:
        preferred_zones = [z.lower() for z in rules["agro_climatic_zones"]]
        matched_zone = False
        for pz in preferred_zones:
            if pz in zone or zone in pz:
                matched_zone = True
                break
        if matched_zone:
            matching.append(f"Indigenous to the {telemetry.get('agro_climatic_zone')} agro-climatic zone.")
            score_offset += 10
        else:
            conflicting.append(f"Typically grown in different agro-climatic zones ({', '.join(rules['agro_climatic_zones'])}).")
            score_offset -= 5

    # 3. Irrigation Type Compatibility
    irr_type = str(telemetry.get("irrigation_type", "")).lower()
    if irr_type:
        preferred_irrs = [i.lower() for i in rules["irrigation_preference"]]
        matched_irr = False
        for pi in preferred_irrs:
            if pi in irr_type or irr_type in pi:
                matched_irr = True
                break
        if matched_irr:
            matching.append(f"Favorable matches with {telemetry.get('irrigation_type')} practices.")
            score_offset += 10
        else:
            conflicting.append(f"Crop prefers {', '.join(rules['irrigation_preference'])} over {telemetry.get('irrigation_type')}.")
            score_offset -= 10
            
    # 4. Nutrient Demand Realism
    # If crop demands high Nitrogen but soil is severely depleted (N < 30): penalty
    # If crop is groundnut (N-fixer) and N is depleted: bonus (reconstructed logic)
    n_val = float(telemetry.get("N", 50))
    n_demand = rules["demands"].get("N", "medium")
    
    if rules.get("nitrogen_fixer") and n_val < NUTRIENT_LEVELS["N"]["low"]:
        matching.append("Acts as a key Nitrogen-fixer to naturally replenish depleted soils.")
        score_offset += 20
    elif n_demand == "high" and n_val < NUTRIENT_LEVELS["N"]["low"]:
        conflicting.append("High Nitrogen consumer; current soil profile displays severe Nitrogen depletion.")
        score_offset -= 15
    elif n_demand == "high" and n_val > NUTRIENT_LEVELS["N"]["high"]:
        matching.append("High Nitrogen consumer; matches rich soil Nitrogen profiles perfectly.")
        score_offset += 10

    # 5. Sustainability Bonus
    sustainability_bonus = int(rules["sustainability_score"] / 10)
    
    return {
        "score_offset": score_offset,
        "matching_factors": matching,
        "conflicting_factors": conflicting,
        "sustainability_bonus": sustainability_bonus,
        "sustainability_factors": rules["sustainability_factors"]
    }
