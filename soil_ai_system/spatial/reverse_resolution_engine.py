import os
import json
import math
from spatial.services.spatial_cache import load_geojson

def point_in_polygon(point, polygon):
    """
    Ray-casting algorithm to check if a point is inside a polygon.
    point: [lon, lat]
    polygon: List of [lon, lat] coordinates forming the polygon.
    """
    x, y = point
    inside = False
    n = len(polygon)
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def resolve_location(lat: float, lon: float) -> dict:
    """
    Resolves a latitude and longitude into a structured administrative hierarchy.
    Uses local GeoJSON boundary matching and nearest-neighbor fallback.
    """
    geojson = load_geojson()
    if not geojson:
        return {}
        
    point = [lon, lat]
    closest_feature = None
    min_dist = float('inf')
    
    for feature in geojson.get("features", []):
        geom_type = feature.get("geometry", {}).get("type")
        coords = feature.get("geometry", {}).get("coordinates", [])
        
        if geom_type == "Polygon":
            poly = coords[0]
            if point_in_polygon(point, poly):
                return feature.get("properties", {})
            
            # For nearest neighbor fallback, use the first point of the polygon
            if poly:
                dist = distance(point, poly[0])
                if dist < min_dist:
                    min_dist = dist
                    closest_feature = feature
                    
        elif geom_type == "MultiPolygon":
            for poly_wrapper in coords:
                poly = poly_wrapper[0]
                if point_in_polygon(point, poly):
                    return feature.get("properties", {})
                
                if poly:
                    dist = distance(point, poly[0])
                    if dist < min_dist:
                        min_dist = dist
                        closest_feature = feature
    
    # Fallback to nearest neighbor if point is not inside any polygon
    if closest_feature and min_dist < 2.0:  # arbitrary reasonable distance threshold in degrees
        return closest_feature.get("properties", {})
        
    return {}
