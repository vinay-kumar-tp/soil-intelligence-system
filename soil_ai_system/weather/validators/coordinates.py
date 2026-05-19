"""Coordinate validation service for Weather API requests.
"""

def validate_coordinates(lat: float, lon: float) -> bool:
    """Validates that latitude and longitude are within standard physical boundaries."""
    try:
        lat_f = float(lat)
        lon_f = float(lon)
        return (-90.0 <= lat_f <= 90.0) and (-180.0 <= lon_f <= 180.0)
    except (ValueError, TypeError):
        return False
