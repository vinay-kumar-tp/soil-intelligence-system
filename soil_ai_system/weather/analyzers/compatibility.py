"""Environmental Compatibility & Reality Validation Engine.

Analyzes alignment between user-submitted telemetry and live weather data.
"""
from typing import Dict, Any, List

def analyze_environmental_compatibility(
    user_inputs: Dict[str, Any],
    weather_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyzes real-time environmental compatibility, crop alignment, and risks."""
    # Fallback to defaults if weather_data has error
    if weather_data.get("status") == "error":
        return {
            "alignment_score": 100,
            "realism_confidence": "HIGH (Fallback Mode)",
            "consistency_warnings": ["Live weather API temporarily offline. Standing by on manual entries."],
            "risks": {
                "drought_risk": "Low",
                "flood_sensitivity": "Low",
                "heat_stress": "Low",
                "water_stress": "Low"
            },
            "narratives": ["Awaiting live environmental telemetry link..."],
            "climate_badges": ["Sensor Normal"]
        }

    # Extract live data
    live_temp = weather_data.get("temperature", 25.0)
    live_hum = weather_data.get("humidity", 50.0)
    live_rain = weather_data.get("rainfall", 0.0)
    wind_speed = weather_data.get("wind_speed", 0.0)
    condition = weather_data.get("weather_condition", "Clear sky")
    forecast = weather_data.get("forecast", {})
    precip_7d = forecast.get("predicted_precipitation_sum_7d", 0.0)

    # Extract user input data
    user_temp = float(user_inputs.get("temperature", live_temp))
    user_hum = float(user_inputs.get("humidity", live_hum))
    user_rain = float(user_inputs.get("rainfall", live_rain))
    irrigation_type = user_inputs.get("irrigation_type", "rain-fed").lower()

    # --- 1. Reality Validation Scorer ---
    temp_diff = abs(user_temp - live_temp)
    hum_diff = abs(user_hum - live_hum)
    
    # Calculate penalty
    temp_penalty = temp_diff * 4.5
    hum_penalty = hum_diff * 1.2
    
    alignment_score = max(5, min(100, int(100 - temp_penalty - hum_penalty)))
    
    warnings = []
    if temp_diff > 12.0:
        warnings.append(f"Temperature anomaly: Local sensors read {live_temp:.1f}°C but manual input is {user_temp:.1f}°C.")
    if hum_diff > 35.0:
        warnings.append(f"Humidity divergence: Local relative humidity is {live_hum:.1f}% vs manual {user_hum:.1f}%.")
    if user_rain > 1200.0 and precip_7d < 10.0 and live_rain == 0:
        warnings.append("Water budget mismatch: Heavy rain context specified but local forecast is dry (<10mm next 7 days).")
        
    if alignment_score >= 80:
        realism_tier = "HIGH CONSISTENCY"
    elif alignment_score >= 50:
        realism_tier = "MODERATE DIVERGENCE"
    else:
        realism_tier = "CRITICAL ANOMALY"
        warnings.insert(0, "User inputs differ significantly from current local environmental conditions.")

    # --- 2. Live Environmental Risk Analysis ---
    drought_risk = "Low"
    flood_risk = "Low"
    heat_stress = "Low"
    water_stress = "Low"
    instability_warnings = []
    
    # Drought Risk Logic
    if live_temp > 35.0 and live_hum < 30.0:
        drought_risk = "High"
        water_stress = "High"
    elif live_temp > 30.0 and live_hum < 45.0:
        drought_risk = "Medium"
        water_stress = "Medium"
        
    if irrigation_type == "rain-fed" and drought_risk != "Low":
        water_stress = "High"
        instability_warnings.append("High drought vulnerability observed under rain-fed crop configurations.")

    # Flood Risk Logic
    if live_rain > 20.0 or precip_7d > 120.0:
        flood_risk = "High"
        instability_warnings.append("Imminent heavy precipitation forecast indicates potential drainage overflow risks.")
    elif live_rain > 5.0 or precip_7d > 60.0:
        flood_risk = "Medium"

    # Heat Stress Logic
    if live_temp > 40.0:
        heat_stress = "High"
        instability_warnings.append("Extreme ambient heat stress limits photosynthesis rates.")
    elif live_temp > 34.0:
        heat_stress = "Medium"

    # --- 3. Dynamic Narrative & Compatibility Generation ---
    narratives = []
    climate_badges = []
    
    if live_hum > 75.0:
        narratives.append("Current atmospheric humidity strongly favors high-transpiration crops like paddy or jute.")
        climate_badges.append("Humid/Wet Zone")
    elif live_hum < 40.0:
        narratives.append("Dry atmospheric conditions reduce confidence for water-intensive surface crops.")
        climate_badges.append("Dry/Arid Zone")
    else:
        narratives.append("Balanced ambient humidity supports standard seasonal transpiration schedules.")
        climate_badges.append("Optimal Vapor Profile")
        
    if precip_7d > 50.0:
        narratives.append("Rainfall forecast indicates robust water compatibility for high-hydration crop cycles.")
    elif precip_7d < 10.0:
        narratives.append("Near-term forecast indicates minimal rainfall; irrigation schedule adjustments advised.")

    if wind_speed > 25.0:
        instability_warnings.append(f"High wind speeds ({wind_speed:.1f} km/h) may trigger mechanical lodging in tall crops.")

    return {
        "alignment_score": alignment_score,
        "realism_confidence": realism_tier,
        "consistency_warnings": warnings,
        "risks": {
            "drought_risk": drought_risk,
            "flood_sensitivity": flood_risk,
            "heat_stress": heat_stress,
            "water_stress": water_stress,
            "instability_warnings": instability_warnings
        },
        "narratives": narratives,
        "climate_badges": climate_badges
    }
