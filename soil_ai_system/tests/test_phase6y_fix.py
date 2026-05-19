import unittest
from spatial.reverse_resolution_engine import resolve_location, point_in_polygon

class TestPhase6YFix(unittest.TestCase):
    def test_point_in_polygon(self):
        poly = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]
        self.assertTrue(point_in_polygon([5, 5], poly))
        self.assertFalse(point_in_polygon([15, 15], poly))

    def test_reverse_resolution_engine(self):
        # We know mock_regions.geojson has a polygon for Thiruvalanjuli
        # Coordinates: [[[79.37, 10.95], [79.4, 10.95], [79.4, 10.98], [79.37, 10.98], [79.37, 10.95]]]
        # Center approx: 79.385, 10.965
        lat, lon = 10.965, 79.385
        res = resolve_location(lat, lon)
        
        # Verify it resolves locally
        self.assertEqual(res.get("state"), "Tamil Nadu")
        self.assertEqual(res.get("district"), "Thanjavur")
        self.assertEqual(res.get("taluk"), "Data Missing")
        self.assertEqual(res.get("village"), "Data Missing")

    def test_reverse_resolution_fallback(self):
        # Coordinates far away
        lat, lon = 50.0, 50.0
        res = resolve_location(lat, lon)
        self.assertEqual(res, {})

if __name__ == '__main__':
    unittest.main()
