"""Phase 6 - Agronomic Intelligence Unit Tests.

Validates confidence tiers, narrative composition, prioritized actions,
reasoning stability, and comparative crop output integrity.
"""

import unittest
from inference.decision_support import (
    generate_decision_support, 
    analyze_crop_suitability,
    NUTRIENT_THRESHOLDS
)

class TestAgronomicDecisionSupport(unittest.TestCase):
    
    def setUp(self):
        # Sample standard inputs
        self.balanced_inputs = {
            "N": 90, "P": 42, "K": 43,
            "temperature": 25.0, "humidity": 82.0,
            "ph": 6.5, "rainfall": 200.0
        }
        
        self.highly_acidic_deficient_inputs = {
            "N": 10, "P": 5, "K": 40,
            "temperature": 25.0, "humidity": 82.0,
            "ph": 4.5, "rainfall": 2000.0  # Excessive rain for most, great for rice
        }

        # Mock predictions representing the model output
        self.crop_res = {
            "prediction": "rice",
            "confidence": 0.88,
            "all_probabilities": {
                "rice": 0.88,
                "maize": 0.08,
                "chickpea": 0.04
            }
        }
        
        self.fertility_res = {"prediction": "High"}
        self.deficiency_res = {"prediction": "Balanced"}

    def test_confidence_threshold_mapping(self):
        """Test that confidence mapping maps to correct tiers based on probability."""
        # High Confidence Check (>= 0.75)
        res = generate_decision_support(self.balanced_inputs, self.crop_res, self.fertility_res, self.deficiency_res)
        self.assertEqual(res["confidence"]["tier"], "HIGH CONFIDENCE")
        self.assertIn("strongly matches", res["confidence"]["message"])

        # Moderate Confidence Check (>= 0.40)
        crop_mod = self.crop_res.copy()
        crop_mod["confidence"] = 0.55
        res_mod = generate_decision_support(self.balanced_inputs, crop_mod, self.fertility_res, self.deficiency_res)
        self.assertEqual(res_mod["confidence"]["tier"], "MODERATE CONFIDENCE")
        self.assertIn("partially supports", res_mod["confidence"]["message"])

        # Low Confidence Check (< 0.40)
        crop_low = self.crop_res.copy()
        crop_low["confidence"] = 0.25
        res_low = generate_decision_support(self.balanced_inputs, crop_low, self.fertility_res, self.deficiency_res)
        self.assertEqual(res_low["confidence"]["tier"], "LOW CONFIDENCE")
        self.assertIn("weakly aligned", res_low["confidence"]["message"])

    def test_narrative_validation_rules(self):
        """Verify the narrative engine dynamically describes soil status scientifically."""
        # Highly Acidic deficient soil check
        deficiency_def = {"prediction": "Nitrogen deficient"}
        res = generate_decision_support(
            self.highly_acidic_deficient_inputs, 
            self.crop_res, 
            self.fertility_res, 
            deficiency_def
        )
        narrative = res["narrative"]
        
        # Must flag acidity
        self.assertIn("acidity is elevated", narrative.lower())
        # Must flag deficiency
        self.assertIn("suboptimal levels", narrative.lower())
        self.assertIn("nitrogen", narrative.lower())

        # Healthy check
        res_healthy = generate_decision_support(self.balanced_inputs, self.crop_res, self.fertility_res, self.deficiency_res)
        self.assertIn("ph is balanced", res_healthy["narrative"].lower())
        self.assertIn("macro-nutrient concentrations", res_healthy["narrative"].lower())

    def test_recommendation_priorities_consistency(self):
        """Ensure priority categories output consistent instructions."""
        # Highly acidic soil should trigger High Priority Lime recommendation
        deficiency_def = {"prediction": "Nitrogen deficient"}
        res = generate_decision_support(
            self.highly_acidic_deficient_inputs, 
            self.crop_res, 
            self.fertility_res, 
            deficiency_def
        )
        high_actions = res["prioritized_actions"]["high"]
        
        # Check that Lime correctors and Nitrogen supplement actions exist
        has_lime = any("lime" in act.lower() for act in high_actions)
        has_nitrogen = any("nitrogen" in act.lower() or "urea" in act.lower() for act in high_actions)
        
        self.assertTrue(has_lime, "Acidity corrector failed to register under high priority actions.")
        self.assertTrue(has_nitrogen, "Nitrogen deficiency corrector failed to register under high priority actions.")

    def test_comparative_crop_index_bounds(self):
        """Validate comparative Top-3 crop grids return clean scientific advantages/risks."""
        res = generate_decision_support(self.balanced_inputs, self.crop_res, self.fertility_res, self.deficiency_res)
        top_k = res["top_k_crops"]
        
        self.assertEqual(len(top_k), 3, "Comparative crops index must return exactly 3 candidates.")
        for crop in top_k:
            self.assertIn("crop", crop)
            self.assertIn("suitability_score", crop)
            self.assertTrue(0 <= crop["suitability_score"] <= 100)
            self.assertGreater(len(crop["advantages"]), 0)
            self.assertGreater(len(crop["risks"]), 0)
            self.assertGreater(len(crop["limiting_factors"]), 0)

    def test_reasoning_stability(self):
        """Ensure extreme values do not cause the reasoning engine to crash."""
        extreme_inputs = {
            "N": 0.0, "P": 0.0, "K": 0.0,
            "temperature": -50.0, "humidity": 0.0,
            "ph": 0.0, "rainfall": 0.0
        }
        crop_res_extreme = self.crop_res.copy()
        crop_res_extreme["all_probabilities"] = {"unknown": 1.0}
        
        try:
            res = generate_decision_support(extreme_inputs, crop_res_extreme, self.fertility_res, self.deficiency_res)
            self.assertIn("hybrid_intelligence_score", res)
        except Exception as e:
            self.fail(f"Reasoning engine crashed on extreme inputs: {e}")

if __name__ == "__main__":
    unittest.main()
