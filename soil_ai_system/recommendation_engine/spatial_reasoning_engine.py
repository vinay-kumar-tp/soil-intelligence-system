"""Phase 6X - Spatial Reasoning Engine.

Interprets location context, adjusts recommendation confidence, adds environmental 
compatibility signals, adds regional crop suitability hints, and generates contextual narratives.
"""
from typing import Dict, Any, List
from configs.spatial_rules import CROP_SUITABILITY_MODIFIERS, IRRIGATION_DEPENDENCE
from configs.agro_climatic_rules import DISTRICT_AGRO_CLIMATIC_MAP, ZONE_MODIFIERS

def apply_spatial_reasoning(
    inputs: Dict[str, Any],
    decision_support: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Augments the decision support outputs with spatial agronomic intelligence.
    Ensures that spatial intelligence modifies recommendations but does NOT replace ML predictions.
    """
    # Extract spatial features
    region_zone = inputs.get("region_zone", "").lower() if inputs.get("region_zone") else ""
    state = inputs.get("state", "")
    district = inputs.get("district", "")
    taluk = inputs.get("taluk", "")
    hobli = inputs.get("hobli", "")
    village = inputs.get("village", "")
    agro_climatic_zone = inputs.get("agro_climatic_zone", "").lower().replace(" ", "_") if inputs.get("agro_climatic_zone") else ""
    irrigation_type = inputs.get("irrigation_type", "").lower() if inputs.get("irrigation_type") else ""
    soil_texture = inputs.get("soil_texture", "").lower() if inputs.get("soil_texture") else ""
    seasonal_context = inputs.get("seasonal_context", "").lower() if inputs.get("seasonal_context") else ""

    spatial_tags = [
        region_zone, agro_climatic_zone, irrigation_type, 
        soil_texture, seasonal_context
    ]
    spatial_tags = [t for t in spatial_tags if t]

    # Enhance top_k_crops in decision_support
    top_crops = decision_support.get("top_k_crops", [])
    for crop_data in top_crops:
        crop_name = crop_data.get("crop", "").lower()
        if crop_name in CROP_SUITABILITY_MODIFIERS:
            rules = CROP_SUITABILITY_MODIFIERS[crop_name]
            
            # Check for bonuses
            bonuses_matched = [tag for tag in spatial_tags if tag in rules["bonuses"]]
            penalties_matched = [tag for tag in spatial_tags if tag in rules["penalties"]]
            
            # Modify suitability score slightly
            score_adjustment = len(bonuses_matched) * 5 - len(penalties_matched) * 5
            new_score = max(10, min(99, crop_data.get("suitability_score", 50) + score_adjustment))
            crop_data["suitability_score"] = new_score
            
            # Generate spatial narratives
            if bonuses_matched:
                crop_data["advantages"].append(rules["narrative"]["bonus"])
            if penalties_matched:
                crop_data["risks"].append(rules["narrative"]["penalty"])

    # Generate region intelligence report
    hierarchy_str = f"{state} > {district} > {taluk}" if taluk else f"{state} ({district})"
    region_intel = {
        "title": f"Region Intelligence: {hierarchy_str}",
        "environmental_context": [],
        "environmental_risks": [],
        "irrigation_assessment": ""
    }
    
    if irrigation_type in IRRIGATION_DEPENDENCE:
        region_intel["irrigation_assessment"] = IRRIGATION_DEPENDENCE[irrigation_type]["description"]
    
    # Simple rule-guided risk indicators based on inputs
    if agro_climatic_zone in ["coastal", "delta_region"]:
        region_intel["environmental_risks"].append("Flood sensitivity and potential salinity concerns.")
        region_intel["environmental_context"].append("High humidity + strong monsoon compatibility.")
    elif agro_climatic_zone in ["arid", "semi_arid", "dry_plains"]:
        region_intel["environmental_risks"].append("Drought risk and water stress observations.")
        region_intel["environmental_context"].append("Dry conditions require efficient water management.")

    # Phase 6Y: Agro-Climatic deep lookup
    if district and district in DISTRICT_AGRO_CLIMATIC_MAP:
        ac_data = DISTRICT_AGRO_CLIMATIC_MAP[district]
        
        # Merge risks
        if ac_data["drought_sensitivity"] == "high":
            region_intel["environmental_risks"].append("District has HIGH drought sensitivity. Water conservation crucial.")
        if ac_data["flood_sensitivity"] == "high":
            region_intel["environmental_risks"].append("District has HIGH flood sensitivity. Ensure proper drainage.")
        if ac_data["salinity_risk"] == "high":
            region_intel["environmental_risks"].append("Coastal/Irrigated salinity risk present.")
            
        # Merge context
        region_intel["environmental_context"].append(f"Rainfall zone: {ac_data['rainfall_zone'].replace('_', ' ').capitalize()}.")
        
        # Apply zone modifiers to top crops
        zone = ac_data["zone"]
        if zone in ZONE_MODIFIERS:
            for crop_data in top_crops:
                cname = crop_data.get("crop", "").lower()
                if cname in ZONE_MODIFIERS[zone]["bonuses"]:
                    crop_data["suitability_score"] = min(99, crop_data.get("suitability_score", 50) + 10)
                    crop_data["advantages"].append(f"District agro-climatic zone ({zone}) strongly favors this crop.")
                if cname in ZONE_MODIFIERS[zone]["penalties"]:
                    crop_data["suitability_score"] = max(10, crop_data.get("suitability_score", 50) - 10)
                    crop_data["risks"].append(f"District agro-climatic zone ({zone}) limits yield potential for this crop.")

    # Deduplicate lists
    region_intel["environmental_risks"] = list(set(region_intel["environmental_risks"]))
    region_intel["environmental_context"] = list(set(region_intel["environmental_context"]))

    # Append spatial narrative to main narrative
    spatial_narrative_parts = []
    if region_intel["environmental_context"]:
        spatial_narrative_parts.append(" ".join(region_intel["environmental_context"]))
    if region_intel["irrigation_assessment"]:
        spatial_narrative_parts.append(region_intel["irrigation_assessment"])
    
    if spatial_narrative_parts:
        decision_support["narrative"] = decision_support.get("narrative", "") + " " + " ".join(spatial_narrative_parts)

    decision_support["region_intelligence"] = region_intel
    
    return decision_support
