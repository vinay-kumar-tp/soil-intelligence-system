import json
import os
from functools import lru_cache

GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "..", "geojson", "real_india_districts.geojson")

@lru_cache(maxsize=1)
def load_geojson():
    if not os.path.exists(GEOJSON_PATH):
        return None
    with open(GEOJSON_PATH, "r") as f:
        return json.load(f)

def clear_spatial_cache():
    load_geojson.cache_clear()
