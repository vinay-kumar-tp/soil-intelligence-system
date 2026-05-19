import sqlite3
import os
from functools import lru_cache

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "hierarchy", "india_admin.db")

@lru_cache(maxsize=1)
def get_all_states():
    if not os.path.exists(DB_PATH):
        return []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM states ORDER BY name")
        return [row[0] for row in cursor.fetchall()]

@lru_cache(maxsize=128)
def get_districts(state_name):
    if not os.path.exists(DB_PATH) or not state_name:
        return []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''SELECT d.name FROM districts d 
                          JOIN states s ON d.state_id = s.id 
                          WHERE s.name = ? ORDER BY d.name''', (state_name,))
        return [row[0] for row in cursor.fetchall()]

@lru_cache(maxsize=1024)
def get_taluks(district_name):
    if not os.path.exists(DB_PATH) or not district_name:
        return []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''SELECT t.name FROM taluks t 
                          JOIN districts d ON t.district_id = d.id 
                          WHERE d.name = ? ORDER BY t.name''', (district_name,))
        return [row[0] for row in cursor.fetchall()]

@lru_cache(maxsize=4096)
def get_hoblis(taluk_name):
    if not os.path.exists(DB_PATH) or not taluk_name:
        return []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''SELECT h.name FROM hoblis h 
                          JOIN taluks t ON h.taluk_id = t.id 
                          WHERE t.name = ? ORDER BY h.name''', (taluk_name,))
        return [row[0] for row in cursor.fetchall()]

@lru_cache(maxsize=8192)
def get_villages(hobli_name):
    if not os.path.exists(DB_PATH) or not hobli_name:
        return []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''SELECT v.name FROM villages v 
                          JOIN hoblis h ON v.hobli_id = h.id 
                          WHERE h.name = ? ORDER BY v.name''', (hobli_name,))
        return [row[0] for row in cursor.fetchall()]

def clear_hierarchy_cache():
    get_all_states.cache_clear()
    get_districts.cache_clear()
    get_taluks.cache_clear()
    get_hoblis.cache_clear()
    get_villages.cache_clear()
