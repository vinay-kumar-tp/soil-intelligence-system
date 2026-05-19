"""Phase 6X - Spatial Agronomic Rules Configuration.

Contains hierarchical rules for regions, state mapping, and contextual reasoning.
"""

SPATIAL_HIERARCHY = {
    "southern_india": {
        "states": ["Tamil Nadu", "Karnataka", "Kerala", "Andhra Pradesh"],
    },
    "northern_plains": {
        "states": ["Punjab", "Haryana", "Uttar Pradesh"],
    },
    "western_india": {
        "states": ["Maharashtra", "Gujarat"],
    },
    "eastern_india": {
        "states": ["West Bengal", "Odisha", "Bihar"],
    }
}

STATE_DISTRICT_MAP = {
    "Tamil Nadu": ["Chennai", "Coimbatore", "Salem", "Thanjavur", "Madurai"],
    "Karnataka": ["Mysore", "Belgaum", "Bangalore", "Hubli"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar"]
}

# Regional bonuses and penalties for crops
CROP_SUITABILITY_MODIFIERS = {
    "rice": {
        "bonuses": ["coastal", "delta_region", "high_rainfall", "canal_irrigation", "humid_coastal", "clay", "kharif"],
        "penalties": ["arid", "dry_zone", "drought_prone", "sandy", "rain_fed"],
        "narrative": {
            "bonus": "coastal/delta conditions and high water availability align strongly with paddy cultivation.",
            "penalty": "dry-zone context and lower water availability reduces confidence for rice cultivation."
        }
    },
    "wheat": {
        "bonuses": ["dry_plains", "irrigated_zone", "rabi", "loamy"],
        "penalties": ["flood_sensitive", "high_rainfall", "humid_coastal", "kharif"],
        "narrative": {
            "bonus": "dry-zone context and cooler conditions favor wheat cultivation.",
            "penalty": "wheat is less suited to humid coastal or flood-prone environments."
        }
    },
    "cotton": {
        "bonuses": ["semi_arid", "black soil", "kharif"],
        "penalties": ["high_rainfall", "flood_sensitive", "humid_coastal"],
        "narrative": {
            "bonus": "semi-arid conditions and black soil strongly favor cotton.",
            "penalty": "high rainfall and flood risks are detrimental to cotton bolls."
        }
    },
    "sugarcane": {
        "bonuses": ["canal_irrigation", "high_rainfall", "delta_region", "clay"],
        "penalties": ["drought_prone", "arid", "rain_fed", "sandy"],
        "narrative": {
            "bonus": "abundant water and heavy soils are excellent for sugarcane.",
            "penalty": "water stress and sandy soils severely limit sugarcane yields."
        }
    }
}

IRRIGATION_DEPENDENCE = {
    "rain-fed": {"risk_multiplier": 1.5, "description": "High dependence on monsoon patterns. Vulnerable to drought risk."},
    "canal irrigation": {"risk_multiplier": 0.8, "description": "Stable water supply through canals. May have salinity risks if overused."},
    "borewell": {"risk_multiplier": 1.1, "description": "Moderate risk depending on groundwater table levels."},
    "drip irrigation": {"risk_multiplier": 0.5, "description": "Highly efficient water use, low water stress risk."}
}

ENVIRONMENTAL_RISKS = {
    "drought_prone": "High risk of water stress during critical growth stages.",
    "flood_sensitive": "Susceptible to waterlogging and crop loss during heavy rains.",
    "salinity_risk": "Potential yield reduction due to soil salinity in coastal or intensively irrigated areas.",
    "high_rainfall": "Risk of nutrient leaching and fungal diseases.",
    "humid_coastal": "High humidity may favor certain pests and fungal growth."
}
