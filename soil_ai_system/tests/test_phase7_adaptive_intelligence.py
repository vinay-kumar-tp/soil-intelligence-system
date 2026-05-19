"""Phase 7 - Connected Agricultural Adaptive Intelligence Unit Tests.

Tests knowledge graphs, session memory recorders, adaptive weighting, 
soil health scores, and long-form narrative coherence.
"""
import unittest

from knowledge_graph.graph import evaluate_graph_compatibility, get_crop_relationships
from session_memory.memory_manager import SessionMemoryManager
from adaptive_reasoning.engine import (
    calculate_soil_health_rating,
    compute_agronomic_intelligence_score,
    execute_adaptive_reasoning
)

class TestAdaptiveIntelligence(unittest.TestCase):
    
    def test_knowledge_graph_consistency(self):
        """Verify the knowledge graph schema assertiveness and fields."""
        rice_rel = get_crop_relationships("rice")
        self.assertIsNotNone(rice_rel)
        self.assertEqual(rice_rel["requires_humidity"], "high")
        self.assertIn("clay", rice_rel["prefers_soil"])
        
        groundnut_rel = get_crop_relationships("groundnut")
        self.assertTrue(groundnut_rel["nitrogen_fixer"])
        
        # Non-existing crop
        self.assertIsNone(get_crop_relationships("unknown_crop"))
        
    def test_relationship_reasoning_compatibility(self):
        """Test soil texture and agro-climatic graph edge preference matching."""
        telemetry_match = {
            "soil_texture": "clay loamy",
            "agro_climatic_zone": "Delta Region",
            "irrigation_type": "canal irrigation",
            "N": 90, "P": 40, "K": 40
        }
        res_match = evaluate_graph_compatibility("rice", telemetry_match)
        self.assertGreater(res_match["score_offset"], 0)
        self.assertTrue(any("Prefers clay loamy" in m for m in res_match["matching_factors"]))
        
        # Mismatched texture
        telemetry_conflict = {
            "soil_texture": "sandy dry",
            "agro_climatic_zone": "Semi-Arid Plains",
            "irrigation_type": "rain-fed",
            "N": 20, "P": 40, "K": 40
        }
        res_conflict = evaluate_graph_compatibility("rice", telemetry_conflict)
        self.assertLess(res_conflict["score_offset"], 0)
        self.assertTrue(any("Soil texture" in c for c in res_conflict["conflicting_factors"]))

    def test_session_memory_drift_and_recuperation(self):
        """Verify memory manager isolates recurring deficiency states and climate drifts."""
        mem = SessionMemoryManager(capacity=5)
        
        telemetry_1 = {"N": 50, "P": 40, "K": 40, "temperature": 25.0, "humidity": 60.0, "rainfall": 120.0}
        results_1 = {"predictions": {"crop": {"prediction": "rice"}, "deficiency": {"prediction": "nitrogen"}}}
        
        telemetry_2 = {"N": 48, "P": 40, "K": 40, "temperature": 28.0, "humidity": 45.0, "rainfall": 50.0}
        results_2 = {"predictions": {"crop": {"prediction": "rice"}, "deficiency": {"prediction": "nitrogen"}}}
        
        mem.add_entry(telemetry_1, results_1)
        mem.add_entry(telemetry_2, results_2)
        
        # Deficiency check
        recurring = mem.detect_recurring_deficiencies()
        self.assertIn("Nitrogen", recurring)
        
        # Temperature drift check
        drift = mem.detect_environmental_drift()
        self.assertTrue(drift["drought_trend"])
        self.assertEqual(drift["temp_drift"], 3.0)
        self.assertEqual(drift["rainfall_trend"], "drying")
        
    def test_soil_health_calculator(self):
        """Test soil health chemical score boundaries."""
        perfect_soil = {"N": 90, "P": 60, "K": 60, "ph": 6.8}
        depleted_soil = {"N": 10, "P": 10, "K": 10, "ph": 4.5}
        
        score_perfect = calculate_soil_health_rating(perfect_soil)
        score_depleted = calculate_soil_health_rating(depleted_soil)
        
        self.assertEqual(score_perfect, 100) # optimal pH (30) + optimal NPK (70)
        self.assertLess(score_depleted, 50)
        
    def test_adaptive_recommendation_weighting(self):
        """Test that water-intensive crops are dynamically penalized under drought drift."""
        # Standard run: no drift, good matching
        input_data = {
            "soil_texture": "clay",
            "agro_climatic_zone": "Delta",
            "irrigation_type": "canal irrigation",
            "N": 80, "P": 40, "K": 40, "ph": 6.5
        }
        base_predictions = {
            "crop": {"prediction": "rice", "confidence": 90.0},
            "fertility": {"prediction": "high"},
            "deficiency": {"prediction": "healthy"}
        }
        weather_intel = {
            "status": "success",
            "compatibility": {
                "alignment_score": 95.0,
                "risks": {"drought_risk": "Low", "flood_sensitivity": "Low", "heat_stress": "Low"}
            }
        }
        
        # Clear memory
        from session_memory.memory_manager import global_session_memory
        global_session_memory.clear()
        
        # First execution (no drift)
        res_normal = execute_adaptive_reasoning(input_data, base_predictions, weather_intel)
        self.assertGreaterEqual(res_normal["intelligence_score"], 80)
        
        # Simulate drought drift in memory
        global_session_memory.add_entry({"temperature": 24.0, "humidity": 65.0, "rainfall": 150.0}, base_predictions)
        global_session_memory.add_entry({"temperature": 29.0, "humidity": 35.0, "rainfall": 40.0}, base_predictions)
        
        # Second execution (drought drift active)
        res_adapted = execute_adaptive_reasoning(input_data, base_predictions, weather_intel)
        
        # The adaptive reasoning must have detected drought trend, penalizing Rice suitability
        self.assertLess(res_adapted["intelligence_score"], res_normal["intelligence_score"])
        self.assertTrue(any("Drying environmental drift" in n for n in res_adapted["adaptive_weight_notices"]))

    def test_long_form_report_coherence(self):
        """Verify the structured AI agronomic report outputs several coherent paragraphs."""
        input_data = {
            "soil_texture": "clay",
            "agro_climatic_zone": "Delta",
            "irrigation_type": "canal irrigation",
            "N": 20, "P": 40, "K": 40, "ph": 6.5
        }
        base_predictions = {
            "crop": {"prediction": "rice", "confidence": 90.0},
            "fertility": {"prediction": "high"},
            "deficiency": {"prediction": "nitrogen"}
        }
        weather_intel = {
            "status": "success",
            "compatibility": {
                "alignment_score": 90.0,
                "risks": {"drought_risk": "Low", "flood_sensitivity": "Low", "heat_stress": "Low"}
            }
        }
        
        res = execute_adaptive_reasoning(input_data, base_predictions, weather_intel)
        report = res["long_form_report"]
        
        # Must be at least 3 paragraph sections
        self.assertGreaterEqual(len(report), 3)
        self.assertTrue(any("optimal" in p.lower() or "consistent" in p.lower() for p in report))
        self.assertTrue(any("nitrogen" in p.lower() or "recovery" in p.lower() for p in report))
        self.assertTrue(any("sustainability" in p.lower() for p in report))
