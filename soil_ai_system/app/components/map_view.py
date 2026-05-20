import folium
from streamlit_folium import st_folium
import streamlit as st
from spatial.loaders.hierarchy_loader import get_all_states, get_districts, get_taluks, get_hoblis, get_villages
from spatial.services.spatial_cache import load_geojson
from spatial.reverse_resolution_engine import resolve_location

def render_interactive_map():
    st.markdown("<h4 style='color: #2b3a18;'>Map-Based Selection</h4>", unsafe_allow_html=True)
    
    # Initialize map
    lat = st.session_state.get('map_lat', 20.5937)
    lng = st.session_state.get('map_lng', 78.9629)
    zoom = 8 if 'map_lat' in st.session_state else 5
    
    m = folium.Map(location=[lat, lng], zoom_start=zoom, tiles="OpenStreetMap")
    
    # Place marker for current selection
    if 'map_lat' in st.session_state:
        folium.Marker(
            [lat, lng], 
            popup="Selected Context", 
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(m)
    
    geojson_data = load_geojson()
    filtered_geojson = None
    
    # Fast filtering: only render boundaries for the currently selected state.
    # This prevents loading a massive 9.5MB dataset into the browser iframe on every rerun!
    selected_state = st.session_state.get('hierarchy_state')
    if geojson_data and selected_state and selected_state != "Data Missing":
        filtered_features = [
            f for f in geojson_data.get("features", [])
            if f.get("properties", {}).get("state", "").lower() == selected_state.lower()
        ]
        if filtered_features:
            filtered_geojson = {
                "type": "FeatureCollection",
                "features": filtered_features
            }
            
    if filtered_geojson:
        folium.GeoJson(
            filtered_geojson,
            name="District Boundaries",
            tooltip=folium.GeoJsonTooltip(fields=["state", "district"])
        ).add_to(m)
        
    st_data = st_folium(m, height=400, width=700, returned_objects=["last_clicked", "last_active_drawing"])
    
    # Handle click: GeoJSON polygon click or raw map click
    clicked_lat = None
    clicked_lng = None
    resolved_props = None
    
    if st_data:
        # If they clicked directly on a region polygon
        if st_data.get("last_active_drawing"):
            props = st_data["last_active_drawing"].get("properties")
            if props:
                resolved_props = props
                # We can approximate the click from the map center or just use the popup
        
        # If they clicked the raw map
        if st_data.get("last_clicked"):
            click = st_data["last_clicked"]
            clicked_lat = click.get("lat")
            clicked_lng = click.get("lng")
            
    # Priority to actual raw coordinates for marker, then resolve
    if clicked_lat is not None and clicked_lng is not None:
        st.info(f"📍 Detected Click: Lat {clicked_lat:.4f}, Lng {clicked_lng:.4f}")
        
        if st.button("Use Selected Location", type="primary"):
            st.session_state['map_lat'] = clicked_lat
            st.session_state['map_lng'] = clicked_lng
            
            # Local reverse geo-resolution (if polygon properties not already captured)
            if not resolved_props:
                resolved_props = resolve_location(clicked_lat, clicked_lng)
                
            if resolved_props:
                st.session_state['hierarchy_state'] = resolved_props.get("state")
                st.session_state['hierarchy_district'] = resolved_props.get("district")
                st.session_state['hierarchy_taluk'] = resolved_props.get("taluk")
                st.session_state['hierarchy_hobli'] = resolved_props.get("hobli")
                st.session_state['hierarchy_village'] = resolved_props.get("village")
                st.success(f"✅ Location successfully resolved & populated: {resolved_props.get('district')}, {resolved_props.get('state')}")
                st.experimental_rerun()
            else:
                st.warning("⚠️ Location not found in local hierarchy boundary database.")
    elif resolved_props:
        # Clicked polygon but no raw coordinate (some st_folium versions)
        st.info(f"📍 Detected Region: {resolved_props.get('district')}, {resolved_props.get('state')}")
        if st.button("Use Selected Region", type="primary"):
            st.session_state['hierarchy_state'] = resolved_props.get("state")
            st.session_state['hierarchy_district'] = resolved_props.get("district")
            st.success(f"✅ Region successfully populated!")
            st.experimental_rerun()

def _reset_children(level):
    if level == 'state':
        st.session_state.hierarchy_district = "Data Missing"
        st.session_state.hierarchy_taluk = "Data Missing"
        st.session_state.hierarchy_hobli = "Data Missing"
        st.session_state.hierarchy_village = "Data Missing"
        if 'dist_select' in st.session_state: del st.session_state['dist_select']
        if 'taluk_select' in st.session_state: del st.session_state['taluk_select']
        if 'hobli_select' in st.session_state: del st.session_state['hobli_select']
        if 'village_select' in st.session_state: del st.session_state['village_select']
    elif level == 'district':
        st.session_state.hierarchy_taluk = "Data Missing"
        st.session_state.hierarchy_hobli = "Data Missing"
        st.session_state.hierarchy_village = "Data Missing"
        if 'taluk_select' in st.session_state: del st.session_state['taluk_select']
        if 'hobli_select' in st.session_state: del st.session_state['hobli_select']
        if 'village_select' in st.session_state: del st.session_state['village_select']
    elif level == 'taluk':
        st.session_state.hierarchy_hobli = "Data Missing"
        st.session_state.hierarchy_village = "Data Missing"
        if 'hobli_select' in st.session_state: del st.session_state['hobli_select']
        if 'village_select' in st.session_state: del st.session_state['village_select']
    elif level == 'hobli':
        st.session_state.hierarchy_village = "Data Missing"
        if 'village_select' in st.session_state: del st.session_state['village_select']

def render_hierarchy_panel():
    st.markdown("#### Selected Spatial Context")
    
    scol1, scol2, scol3 = st.columns(3)
    
    states = get_all_states()
    if not states:
        states = ["Data Missing"]
        
    # Initialize session state if missing
    if 'hierarchy_state' not in st.session_state:
        st.session_state.hierarchy_state = states[0]
    if 'hierarchy_district' not in st.session_state:
        st.session_state.hierarchy_district = "Data Missing"
    if 'hierarchy_taluk' not in st.session_state:
        st.session_state.hierarchy_taluk = "Data Missing"
    if 'hierarchy_hobli' not in st.session_state:
        st.session_state.hierarchy_hobli = "Data Missing"
    if 'hierarchy_village' not in st.session_state:
        st.session_state.hierarchy_village = "Data Missing"
        
    # Safety sync
    if st.session_state.hierarchy_state not in states and states:
        st.session_state.hierarchy_state = states[0]
        _reset_children('state')

    with scol1:
        sel_state = st.selectbox("State / Union Territory", states, 
                                 index=states.index(st.session_state.hierarchy_state) if st.session_state.hierarchy_state in states else 0,
                                 key="state_select")
        
        if sel_state != st.session_state.hierarchy_state:
            st.session_state.hierarchy_state = sel_state
            _reset_children('state')
            st.experimental_rerun()
                
        districts = get_districts(st.session_state.hierarchy_state)
        if not districts: districts = ["Data Missing"]
        
        # Safety sync
        if st.session_state.hierarchy_district not in districts:
            st.session_state.hierarchy_district = districts[0]
            _reset_children('district')
            
        sel_dist = st.selectbox("District", districts, 
                                index=districts.index(st.session_state.hierarchy_district) if st.session_state.hierarchy_district in districts else 0,
                                key="dist_select")
        if sel_dist != st.session_state.hierarchy_district:
            st.session_state.hierarchy_district = sel_dist
            _reset_children('district')
            st.experimental_rerun()
            
    with scol2:
        taluks = get_taluks(st.session_state.hierarchy_district)
        if not taluks: taluks = ["Data Missing"]
        
        if st.session_state.hierarchy_taluk not in taluks:
            st.session_state.hierarchy_taluk = taluks[0]
            _reset_children('taluk')
            
        sel_taluk = st.selectbox("Taluk / Tehsil", taluks, 
                                 index=taluks.index(st.session_state.hierarchy_taluk) if st.session_state.hierarchy_taluk in taluks else 0,
                                 key="taluk_select")
        if sel_taluk != st.session_state.hierarchy_taluk:
            st.session_state.hierarchy_taluk = sel_taluk
            _reset_children('taluk')
            st.experimental_rerun()
                
        hoblis = get_hoblis(st.session_state.hierarchy_taluk)
        if not hoblis: hoblis = ["Data Missing"]
        
        if st.session_state.hierarchy_hobli not in hoblis:
            st.session_state.hierarchy_hobli = hoblis[0]
            _reset_children('hobli')
            
        sel_hobli = st.selectbox("Hobli / Block", hoblis, 
                                 index=hoblis.index(st.session_state.hierarchy_hobli) if st.session_state.hierarchy_hobli in hoblis else 0,
                                 key="hobli_select")
        if sel_hobli != st.session_state.hierarchy_hobli:
            st.session_state.hierarchy_hobli = sel_hobli
            _reset_children('hobli')
            st.experimental_rerun()
            
    with scol3:
        villages = get_villages(st.session_state.hierarchy_hobli)
        if not villages: villages = ["Data Missing"]
        
        if st.session_state.hierarchy_village not in villages:
            st.session_state.hierarchy_village = villages[0]
            
        sel_village = st.selectbox("Village", villages, 
                                   index=villages.index(st.session_state.hierarchy_village) if st.session_state.hierarchy_village in villages else 0,
                                   key="village_select")
        if sel_village != st.session_state.hierarchy_village:
            st.session_state.hierarchy_village = sel_village
            st.experimental_rerun()

        
        region_zone = st.selectbox("Region Zone", ["southern_india", "northern_plains", "coastal", "semi_arid", "western_ghats", "delta_region", "dry_zone"])

    st.markdown("---")
    
    ecol1, ecol2 = st.columns(2)
    with ecol1:
        agro_climatic_zone = st.selectbox("Agro-Climatic Zone", ["Coastal", "Delta", "Semi-Arid", "Western Ghats", "Dry Plains"])
        irrigation_type = st.selectbox("Irrigation Type", ["Rain-fed", "Canal irrigation", "Borewell", "Drip irrigation"])
    with ecol2:
        seasonal_context = st.selectbox("Seasonal Context", ["Kharif", "Rabi", "Summer"])
        
    # Return payload dict
    return {
        "region_zone": region_zone,
        "state": sel_state if sel_state != "Data Missing" else None,
        "district": sel_dist if sel_dist != "Data Missing" else None,
        "taluk": sel_taluk if sel_taluk != "Data Missing" else None,
        "hobli": sel_hobli if sel_hobli != "Data Missing" else None,
        "village": sel_village if sel_village != "Data Missing" else None,
        "agro_climatic_zone": agro_climatic_zone,
        "irrigation_type": irrigation_type,
        "seasonal_context": seasonal_context
    }
