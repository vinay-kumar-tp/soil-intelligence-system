import os
import json
import sqlite3

def setup_spatial_data():
    base_dir = "spatial"
    dirs = [
        "datasets",
        "hierarchy",
        "geojson",
        "cache",
        "loaders",
        "services"
    ]
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
        
    # 1. Create a SQLite DB for hierarchical lazy loading
    db_path = os.path.join(base_dir, "hierarchy", "india_admin.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''CREATE TABLE IF NOT EXISTS states (id INTEGER PRIMARY KEY, name TEXT UNIQUE)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS districts (id INTEGER PRIMARY KEY, state_id INTEGER, name TEXT, FOREIGN KEY(state_id) REFERENCES states(id))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS taluks (id INTEGER PRIMARY KEY, district_id INTEGER, name TEXT, FOREIGN KEY(district_id) REFERENCES districts(id))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS hoblis (id INTEGER PRIMARY KEY, taluk_id INTEGER, name TEXT, FOREIGN KEY(taluk_id) REFERENCES taluks(id))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS villages (id INTEGER PRIMARY KEY, hobli_id INTEGER, name TEXT, FOREIGN KEY(hobli_id) REFERENCES hoblis(id))''')
    
    # Insert mock data
    cursor.execute("INSERT OR IGNORE INTO states (id, name) VALUES (1, 'Tamil Nadu'), (2, 'Karnataka'), (3, 'Punjab')")
    
    # Districts for TN (1)
    cursor.executemany("INSERT OR IGNORE INTO districts (id, state_id, name) VALUES (?,?,?)", [
        (1, 1, 'Thanjavur'),
        (2, 1, 'Coimbatore')
    ])
    
    # Taluks for Thanjavur (1)
    cursor.executemany("INSERT OR IGNORE INTO taluks (id, district_id, name) VALUES (?,?,?)", [
        (1, 1, 'Kumbakonam'),
        (2, 1, 'Papanasam')
    ])
    
    # Hoblis for Kumbakonam (1)
    cursor.executemany("INSERT OR IGNORE INTO hoblis (id, taluk_id, name) VALUES (?,?,?)", [
        (1, 1, 'Swamimalai'),
        (2, 1, 'Darasuram')
    ])
    
    # Villages for Swamimalai (1)
    cursor.executemany("INSERT OR IGNORE INTO villages (id, hobli_id, name) VALUES (?,?,?)", [
        (1, 1, 'Thiruvalanjuli'),
        (2, 1, 'Baburajapuram')
    ])
    
    # Create indexes for fast lookup
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_district_state ON districts(state_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_taluk_district ON taluks(district_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_hobli_taluk ON hoblis(taluk_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_village_hobli ON villages(hobli_id)")
    
    conn.commit()
    conn.close()
    
    # 2. Create mock GeoJSON for the map
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "state": "Tamil Nadu",
                    "district": "Thanjavur",
                    "taluk": "Kumbakonam",
                    "hobli": "Swamimalai",
                    "village": "Thiruvalanjuli"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[79.37, 10.95], [79.4, 10.95], [79.4, 10.98], [79.37, 10.98], [79.37, 10.95]]]
                }
            }
        ]
    }
    
    with open(os.path.join(base_dir, "geojson", "mock_regions.geojson"), "w") as f:
        json.dump(geojson_data, f)
        
    print("Spatial DB and GeoJSON generated successfully.")

if __name__ == "__main__":
    setup_spatial_data()
