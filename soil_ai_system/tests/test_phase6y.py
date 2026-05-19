import unittest
import os
from spatial.loaders.hierarchy_loader import get_all_states, get_districts, get_taluks, get_hoblis, get_villages
from recommendation_engine.spatial_reasoning_engine import apply_spatial_reasoning
from spatial.services.spatial_cache import load_geojson

class TestPhase6Y(unittest.TestCase):
    def test_lazy_hierarchy_loading(self):
        states = get_all_states()
        self.assertIn("Tamil Nadu", states)
        
        districts = get_districts("Tamil Nadu")
        self.assertIn("Thanjavur", districts)
        
        taluks = get_taluks("Thanjavur")
        self.assertEqual(taluks, [])

    def test_spatial_caching(self):
        info = get_all_states.cache_info()
        # We can't guarantee hits depending on run order, but we can verify it's a cached function
        self.assertIsNotNone(info)

    def test_geojson_loads(self):
        geojson = load_geojson()
        self.assertIsNotNone(geojson)
        self.assertEqual(geojson["type"], "FeatureCollection")

    def test_spatial_reasoning_engine_agro_climatic(self):
        inputs = {
            "region_zone": "southern_india",
            "state": "Tamil Nadu",
            "district": "Thanjavur",
            "taluk": "Kumbakonam",
            "agro_climatic_zone": "delta_region",
            "irrigation_type": "canal irrigation",
            "soil_texture": "clay"
        }
        
        decision_support = {
            "top_k_crops": [{"crop": "Rice", "suitability_score": 60, "advantages": [], "risks": []}],
            "narrative": "Standard analysis."
        }
        
        res = apply_spatial_reasoning(inputs, decision_support)
        rice = res["top_k_crops"][0]
        
        # Thanjavur is in DISTRICT_AGRO_CLIMATIC_MAP as delta_region, which boosts rice
        self.assertTrue(rice["suitability_score"] > 60)
        
        reg_intel = res.get("region_intelligence", {})
        # Should include Thanjavur risk factors like flood sensitivity & salinity risk
        risks = reg_intel.get("environmental_risks", [])
        self.assertTrue(any("flood" in r.lower() for r in risks))
        self.assertTrue(any("salinity" in r.lower() for r in risks))

if __name__ == '__main__':
    unittest.main()
