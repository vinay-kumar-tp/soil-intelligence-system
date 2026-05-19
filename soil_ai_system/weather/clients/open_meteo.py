"""Open-Meteo API Client for Soil AI System.

Handles direct weather queries, network errors, and normalizes responses.
"""
import requests
import logging
from typing import Dict, Any

logger = logging.getLogger("weather.clients.open_meteo")

WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
}

def fetch_live_weather(lat: float, lon: float) -> Dict[str, Any]:
    """Fetches real-time weather and short-term forecast from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto"
    }
    
    # Ignore system proxies to prevent WinError 10061/timeout in sandboxes
    session = requests.Session()
    session.trust_env = False
    
    try:
        response = session.get(url, params=params, timeout=6)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        daily = data.get("daily", {})
        
        weather_code = current.get("weather_code", 0)
        condition = WMO_WEATHER_CODES.get(weather_code, "Unknown")
        
        # Normalize the forecast
        temp_maxs = daily.get("temperature_2m_max", [])
        temp_mins = daily.get("temperature_2m_min", [])
        precip_sums = daily.get("precipitation_sum", [])
        
        avg_temp_max = sum(temp_maxs) / len(temp_maxs) if temp_maxs else current.get("temperature_2m", 25)
        avg_temp_min = sum(temp_mins) / len(temp_mins) if temp_mins else current.get("temperature_2m", 25)
        total_precip_sum = sum(precip_sums) if precip_sums else 0.0
        
        return {
            "status": "success",
            "temperature": current.get("temperature_2m", 25.0),
            "humidity": current.get("relative_humidity_2m", 50.0),
            "rainfall": current.get("precipitation", 0.0),
            "wind_speed": current.get("wind_speed_10m", 0.0),
            "weather_condition": condition,
            "forecast": {
                "avg_temp_max": round(avg_temp_max, 1),
                "avg_temp_min": round(avg_temp_min, 1),
                "predicted_precipitation_sum_7d": round(total_precip_sum, 1)
            }
        }
    except Exception as e:
        logger.warning("Failed to fetch weather from Open-Meteo: %s", e)
        return {
            "status": "error",
            "message": str(e)
        }
