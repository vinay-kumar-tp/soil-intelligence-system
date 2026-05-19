import os
import json
import sqlite3
import requests

def fetch_and_setup_real_data():
    base_dir = "spatial"
    geojson_path = os.path.join(base_dir, "geojson", "real_india_districts.geojson")
    db_path = os.path.join(base_dir, "hierarchy", "india_admin.db")
    
    print("Downloading India Districts GeoJSON (approx 9.5MB)...")
    url = "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    # 1. Transform GeoJSON to use 'state' and 'district' keys to match our codebase
    states_dict = {}
    districts_dict = {}
    
    print("Processing features...")
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        state = props.get("NAME_1", "Unknown")
        district = props.get("NAME_2", "Unknown")
        
        # Override properties for our frontend tooltip & reverse resolution engine
        feature["properties"] = {
            "state": state,
            "district": district,
            "taluk": "Data Missing",
            "hobli": "Data Missing",
            "village": "Data Missing"
        }
        
        if state not in states_dict:
            states_dict[state] = set()
        states_dict[state].add(district)
        
    print("Saving normalized GeoJSON...")
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
        
    # 2. Update SQLite Database
    print("Updating SQLite Hierarchy Database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear existing states and districts
    cursor.execute("DELETE FROM states")
    cursor.execute("DELETE FROM districts")
    # Also clear children so there's no orphaned dummy data
    cursor.execute("DELETE FROM taluks")
    cursor.execute("DELETE FROM hoblis")
    cursor.execute("DELETE FROM villages")
    
    state_id = 1
    dist_id = 1
    
    for state, districts in states_dict.items():
        cursor.execute("INSERT INTO states (id, name) VALUES (?, ?)", (state_id, state))
        
        for district in districts:
            cursor.execute("INSERT INTO districts (id, state_id, name) VALUES (?, ?, ?)", (dist_id, state_id, district))
            dist_id += 1
            
        state_id += 1
        
    conn.commit()
    conn.close()
    
    print("Database and GeoJSON updated successfully with real data!")

if __name__ == "__main__":
    fetch_and_setup_real_data()
