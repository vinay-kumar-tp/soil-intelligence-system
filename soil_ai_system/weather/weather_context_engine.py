"""Phase 6Z - Weather Context Orchestration Engine.

Orchestrates fetching, caching, validating, and analyzing real-time 
environmental and agricultural weather variables.
"""
import logging
from typing import Dict, Any

from weather.validators.coordinates import validate_coordinates
from weather.cache.weather_cache import global_weather_cache
from weather.clients.open_meteo import fetch_live_weather
from weather.analyzers.compatibility import analyze_environmental_compatibility

logger = logging.getLogger("weather.weather_context_engine")

def get_live_weather_context(
    lat: float,
    lon: float,
    user_inputs: Dict[str, Any]
) -> Dict[str, Any]:
    """Orchestrates coordinate-based weather fetching, caching, and reality validation.
    
    Args:
        lat: Latitude of target location.
        lon: Longitude of target location.
        user_inputs: Dictionary of user-submitted physical telemetry (temp, hum, ph, rain, etc.)
        
    Returns:
        A rich environmental context report dictionary.
    """
    # 1. Validate coordinates
    if not validate_coordinates(lat, lon):
        logger.warning("Invalid coordinates passed to weather engine: (%s, %s)", lat, lon)
        return {
            "status": "error",
            "message": "Coordinates are invalid or outside boundary limits.",
            "weather": {},
            "compatibility": analyze_environmental_compatibility(user_inputs, {"status": "error"})
        }
        
    # 2. Check Cache
    cached_weather = global_weather_cache.get(lat, lon)
    if cached_weather:
        logger.info("Weather cache hit for key context (%s, %s)", lat, lon)
        weather_data = cached_weather
    else:
        logger.info("Weather cache miss. Querying Open-Meteo API for (%s, %s)", lat, lon)
        weather_data = fetch_live_weather(lat, lon)
        if weather_data.get("status") == "success":
            global_weather_cache.set(lat, lon, weather_data)
            
    # 3. Analyze Compatibility & reality scores
    compatibility = analyze_environmental_compatibility(user_inputs, weather_data)
    
    return {
        "status": "success" if weather_data.get("status") == "success" else "fallback",
        "weather": weather_data,
        "compatibility": compatibility
    }
