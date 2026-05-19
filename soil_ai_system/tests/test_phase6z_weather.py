"""Phase 6Z Weather & Environmental Context Orchestration Unit Tests.

Tests caching, validators, API clients, reasoning, alignment scores, 
and graceful degradation under API failure modes.
"""
import unittest
from unittest.mock import patch, MagicMock
import time

from weather.validators.coordinates import validate_coordinates
from weather.cache.weather_cache import WeatherCache, global_weather_cache
from weather.clients.open_meteo import fetch_live_weather
from weather.analyzers.compatibility import analyze_environmental_compatibility
from weather.weather_context_engine import get_live_weather_context

class TestWeatherIntelligence(unittest.TestCase):
    
    def setUp(self):
        global_weather_cache.clear()
        
    def test_coordinate_validation(self):
        """Verify boundary validations for geographic coordinates."""
        self.assertTrue(validate_coordinates(12.9716, 77.5946)) # Bangalore
        self.assertTrue(validate_coordinates(-90.0, 180.0))
        self.assertFalse(validate_coordinates(95.0, 77.0)) # Out of bounds lat
        self.assertFalse(validate_coordinates(12.0, -190.0)) # Out of bounds lon
        self.assertFalse(validate_coordinates("invalid", 77.0))
        
    def test_weather_cache_functionality(self):
        """Test cache rounding, hit/miss patterns, and TTL expiration."""
        cache = WeatherCache(ttl_seconds=1)
        
        # Rounding check: 12.9716 and 12.9749 both round to 12.97 under 2-decimal rounding
        cache.set(12.9716, 77.5946, {"temp": 28})
        self.assertEqual(cache.get(12.9749, 77.5922), {"temp": 28}) # HIT due to proximity rounding
        
        # Miss check
        self.assertIsNone(cache.get(13.5, 77.5))
        
        # TTL check
        time.sleep(1.1)
        self.assertIsNone(cache.get(12.9716, 77.5946)) # Expired
        
    def test_environmental_alignment_logic(self):
        """Test temperature/humidity divergence penalties and warning prompts."""
        user_inputs = {
            "temperature": 25.0,
            "humidity": 40.0,
            "rainfall": 100.0,
            "irrigation_type": "rain-fed"
        }
        
        # Scenario A: Perfect consistency
        weather_perfect = {
            "status": "success",
            "temperature": 25.0,
            "humidity": 40.0,
            "rainfall": 0.0,
            "weather_condition": "Clear sky",
            "forecast": {"predicted_precipitation_sum_7d": 0.0}
        }
        res_perfect = analyze_environmental_compatibility(user_inputs, weather_perfect)
        self.assertEqual(res_perfect["alignment_score"], 100)
        self.assertEqual(res_perfect["realism_confidence"], "HIGH CONSISTENCY")
        
        # Scenario B: High divergence (Critical anomaly)
        weather_divergent = {
            "status": "success",
            "temperature": 38.0, # 13 degrees difference
            "humidity": 80.0, # 40 percent difference
            "rainfall": 0.0,
            "weather_condition": "Heavy rain",
            "forecast": {"predicted_precipitation_sum_7d": 150.0}
        }
        res_divergent = analyze_environmental_compatibility(user_inputs, weather_divergent)
        self.assertLess(res_divergent["alignment_score"], 50)
        self.assertEqual(res_divergent["realism_confidence"], "CRITICAL ANOMALY")
        self.assertTrue(any("divergence" in w.lower() or "anomaly" in w.lower() or "differ" in w.lower() for w in res_divergent["consistency_warnings"]))

    def test_environmental_risk_matrix(self):
        """Test risk matrix evaluation based on live weather data."""
        user_inputs = {"temperature": 25.0, "humidity": 50.0}
        
        # Scenario A: High heat and dry air (Drought & Water stress)
        weather_drought = {
            "status": "success",
            "temperature": 39.0,
            "humidity": 20.0,
            "rainfall": 0.0,
            "weather_condition": "Clear sky",
            "forecast": {"predicted_precipitation_sum_7d": 2.0}
        }
        res = analyze_environmental_compatibility(user_inputs, weather_drought)
        self.assertEqual(res["risks"]["drought_risk"], "High")
        self.assertEqual(res["risks"]["heat_stress"], "Medium")
        
        # Scenario B: Flood risk (Extreme rainfall next 7 days)
        weather_flood = {
            "status": "success",
            "temperature": 22.0,
            "humidity": 90.0,
            "rainfall": 25.0,
            "weather_condition": "Heavy rain",
            "forecast": {"predicted_precipitation_sum_7d": 140.0}
        }
        res2 = analyze_environmental_compatibility(user_inputs, weather_flood)
        self.assertEqual(res2["risks"]["flood_sensitivity"], "High")

    @patch("weather.clients.open_meteo.requests.Session.get")
    def test_api_success_and_caching(self, mock_get):
        """Verify successful client query parses correctly and populates the cache."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "current": {
                "temperature_2m": 30.0,
                "relative_humidity_2m": 60.0,
                "precipitation": 0.0,
                "wind_speed_10m": 12.0,
                "weather_code": 0
            },
            "daily": {
                "temperature_2m_max": [32.0],
                "temperature_2m_min": [28.0],
                "precipitation_sum": [5.0]
            }
        }
        mock_get.return_value = mock_response
        
        # Run service orchestrator
        user_inputs = {"temperature": 30.0, "humidity": 60.0}
        result = get_live_weather_context(12.97, 77.59, user_inputs)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["weather"]["temperature"], 30.0)
        self.assertEqual(result["compatibility"]["alignment_score"], 100)
        
        # Verify cache was populated
        cached = global_weather_cache.get(12.97, 77.59)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["temperature"], 30.0)
        
    @patch("weather.weather_context_engine.fetch_live_weather")
    def test_graceful_degradation_on_api_failure(self, mock_fetch):
        """Ensure system falls back gracefully when API returns error status."""
        mock_fetch.return_value = {"status": "error", "message": "Connection timeout"}
        user_inputs = {"temperature": 25.0, "humidity": 50.0}
        
        # Call weather context engine under failure mode
        result = get_live_weather_context(12.97, 77.59, user_inputs)
        
        # Graceful degradation check
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["compatibility"]["alignment_score"], 100)
        self.assertTrue("API temporarily offline" in result["compatibility"]["consistency_warnings"][0])
