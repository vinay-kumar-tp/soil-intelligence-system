"""TTL-based Weather Cache for coordinates.

Prevents duplicated queries to the Open-Meteo API.
"""
import time
from typing import Dict, Any, Optional

class WeatherCache:
    def __init__(self, ttl_seconds: int = 1800):
        self.ttl = ttl_seconds
        self.store: Dict[str, Dict[str, Any]] = {}
        
    def _make_key(self, lat: float, lon: float) -> str:
        # Round to 2 decimal places (roughly ~1.1km grid accuracy) to avoid micro-variations
        return f"{round(lat, 2):.2f}_{round(lon, 2):.2f}"
        
    def get(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        key = self._make_key(lat, lon)
        if key in self.store:
            entry = self.store[key]
            if time.time() - entry["timestamp"] <= self.ttl:
                return entry["data"]
            else:
                # Expired
                del self.store[key]
        return None
        
    def set(self, lat: float, lon: float, data: Dict[str, Any]) -> None:
        key = self._make_key(lat, lon)
        self.store[key] = {
            "timestamp": time.time(),
            "data": data
        }
        
    def clear(self) -> None:
        self.store.clear()

global_weather_cache = WeatherCache()
