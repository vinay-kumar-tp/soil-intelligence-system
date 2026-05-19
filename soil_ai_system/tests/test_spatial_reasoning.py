import unittest
from recommendation_engine.spatial_reasoning_engine import apply_spatial_reasoning

class TestSpatialReasoning(unittest.TestCase):
    def test_spatial_reasoning_enhances_score(self):
        inputs = {
            "region_zone": "coastal",
            "state": "Tamil Nadu",
            "district": "Chennai",
            "agro_climatic_zone": "coastal",
            "irrigation_type": "canal_irrigation",
            "soil_texture": "clay",
            "seasonal_context": "kharif"
        }
        
        decision_support = {
            "top_k_crops": [
                {
                    "crop": "Rice",
                    "suitability_score": 50,
                    "advantages": [],
                    "risks": []
                }
            ],
            "narrative": "Base narrative."
        }
        
        res = apply_spatial_reasoning(inputs, decision_support)
        rice_data = res["top_k_crops"][0]
        
        # Should be > 50 because of bonuses (coastal, clay, kharif)
        self.assertTrue(rice_data["suitability_score"] > 50)
        self.assertIn("Region Intelligence", res.get("region_intelligence", {}).get("title", ""))
        self.assertIn("Base narrative.", res["narrative"])

    def test_deterministic_reasoning(self):
        inputs = {
            "agro_climatic_zone": "semi_arid",
            "irrigation_type": "rain-fed"
        }
        ds1 = apply_spatial_reasoning(inputs, {"top_k_crops": []})
        ds2 = apply_spatial_reasoning(inputs, {"top_k_crops": []})
        self.assertEqual(ds1, ds2)
        
    def test_environmental_risk_tests(self):
        inputs = {
            "agro_climatic_zone": "arid"
        }
        res = apply_spatial_reasoning(inputs, {})
        risks = res.get("region_intelligence", {}).get("environmental_risks", [])
        self.assertTrue(any("Drought risk" in r for r in risks))

if __name__ == '__main__':
    unittest.main()
