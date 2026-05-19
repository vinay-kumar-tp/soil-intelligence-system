"""Agro-Climatic Intelligence Rules

Maps spatial entities (districts/zones) to their environmental risk factors,
rainfall profiles, and inherent crop tendencies.
"""

DISTRICT_AGRO_CLIMATIC_MAP = {
    "Thanjavur": {
        "zone": "delta_region",
        "rainfall_zone": "high_rainfall",
        "drought_sensitivity": "low",
        "humidity_profile": "humid_coastal",
        "irrigation_dependency": "canal_irrigation",
        "salinity_risk": "high",
        "flood_sensitivity": "high",
        "crop_tendencies": ["rice", "sugarcane", "banana"]
    },
    "Coimbatore": {
        "zone": "semi_arid",
        "rainfall_zone": "low_rainfall",
        "drought_sensitivity": "high",
        "humidity_profile": "dry",
        "irrigation_dependency": "borewell",
        "salinity_risk": "low",
        "flood_sensitivity": "low",
        "crop_tendencies": ["cotton", "maize", "groundnut"]
    }
}

ZONE_MODIFIERS = {
    "delta_region": {
        "bonuses": ["rice", "sugarcane"],
        "penalties": ["wheat", "cotton"]
    },
    "semi_arid": {
        "bonuses": ["cotton", "jowar", "maize"],
        "penalties": ["rice", "banana"]
    }
}
