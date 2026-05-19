import streamlit as st
import sys
from pathlib import Path
import time

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from frontend.services.api_client import get_system_health, predict_soil
from frontend.visualizations.charts import plot_confidence_gauge, plot_feature_importance
from frontend.components.cards import render_recommendation_card, render_contrastive_panel
from frontend.components.decision_cards import (
    render_confidence_indicator,
    render_hybrid_intelligence_score_gauge,
    render_agronomic_narrative,
    render_prioritized_recommendations,
    render_comparative_crops
)
from frontend.components.map_view import render_interactive_map, render_hierarchy_panel

st.set_page_config(
    page_title="AgroSphere - Precision Agronomic Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Olive Green & White Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Clean Olive Green & White Background */
    .stApp {
        background-color: #fdfdfc;
        background-image: linear-gradient(180deg, #fdfdfc 0%, #f4f7f4 100%);
        color: #2b3a18;
    }
    
    /* Top Navigation Mock */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 0;
        margin-bottom: 3rem;
        border-bottom: 2px solid rgba(85, 107, 47, 0.1);
    }
    .logo { color: #4d7c0f; font-weight: 800; font-size: 1.8rem; display: flex; align-items: center; gap: 10px;}
    .nav-links { display: flex; gap: 30px; color: #4b5563; font-weight: 600; font-size: 1rem;}
    
    /* Hero Section */
    .hero-subtitle {
        color: #65a30d;
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .main-header {
        color: #2b3a18;
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 1.5rem;
        max-width: 800px;
    }
    .hero-desc {
        color: #4b5563;
        font-size: 1.2rem;
        max-width: 650px;
        margin-bottom: 3rem;
        line-height: 1.6;
    }
    
    /* Square Metric Blocks */
    .block-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 25px;
        margin-bottom: 3rem;
    }
    .square-block {
        border-radius: 20px;
        padding: 2.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        box-shadow: 0 15px 35px rgba(85, 107, 47, 0.08);
        transition: transform 0.3s ease;
        height: 250px;
    }
    .square-block:hover { transform: translateY(-8px); box-shadow: 0 20px 40px rgba(85, 107, 47, 0.15); }
    .block-icon {
        background: rgba(255,255,255,0.3);
        border-radius: 50%;
        width: 75px;
        height: 75px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    .block-green { background: linear-gradient(135deg, #65a30d, #4d7c0f); color: white; border: 1px solid #a3e635;}
    .block-brown { background: linear-gradient(135deg, #78350f, #451a03); color: white; border: 1px solid #d97706;}
    .block-tan { background: linear-gradient(135deg, #d4a373, #b59874); color: white; border: 1px solid #fcd34d;}
    
    .metric-title { font-size: 1.1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 5px; opacity: 0.9;}
    .metric-value { font-size: 2.5rem; font-weight: 800; margin: 0; text-transform: capitalize; }
    
    /* Form Area (Clean Light Mode) */
    .form-panel {
        background: #ffffff;
        border: 1px solid rgba(85, 107, 47, 0.15);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 10px 40px rgba(85, 107, 47, 0.05);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #65a30d, #4d7c0f);
        color: white;
        border-radius: 50px;
        padding: 1rem 2.5rem;
        font-weight: 800;
        font-size: 1.1rem;
        border: none;
        width: auto;
        box-shadow: 0 10px 20px rgba(77, 124, 15, 0.25);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4d7c0f, #3f6212);
        color: white;
        transform: scale(1.05);
    }
    
    /* Input Styling */
    .stNumberInput>div>div>input, .stSelectbox>div>div>div { 
        font-size: 1.1rem; 
        padding: 0.6rem; 
        border-radius: 10px; 
        background: #f8fafc; 
        color: #1e293b; 
        border: 1px solid #cbd5e1;
        transition: border 0.3s;
    }
    
    .stNumberInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
        border-color: #65a30d;
        box-shadow: 0 0 0 2px rgba(101, 163, 13, 0.2);
    }
    
    /* Text Overrides for Form */
    .stMarkdown h3, .stMarkdown h4 { color: #2b3a18 !important; font-weight: 800;}
    label { color: #475569 !important; font-weight: 700 !important; font-size: 0.95rem !important; }
</style>
""", unsafe_allow_html=True)

def page_predictive_analysis():
    # Top Navigation Simulation
    st.markdown("""
    <div class="top-nav">
        <div class="logo">🌾 AgroSphere</div>
        <div class="nav-links">
            <span>Home</span>
            <span>About Us</span>
            <span>Precision Services ▾</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-subtitle">HIGH-PERFORMANCE PRECISION AGRONOMY</div>
    <div class="main-header">Soil Intelligence<br>& Crop Suitability</div>
    <div class="hero-desc">Configure your field's soil metrics to generate AI-driven agronomic recommendations. A successful farm utilizes precision data to maximize yield and minimize chemical runoff.</div>
    """, unsafe_allow_html=True)

    # --- Input Dashboard ---
    with st.container():
        st.markdown("<div class='form-panel'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: white; margin-bottom: 20px;'>Field Telemetry Inputs</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_val = st.number_input("Nitrogen (N)", 0.0, 200.0, 90.0)
            temp_val = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0)
        with col2:
            p_val = st.number_input("Phosphorus (P)", 0.0, 200.0, 42.0)
            hum_val = st.number_input("Humidity (%)", 0.0, 100.0, 82.0)
        with col3:
            k_val = st.number_input("Potassium (K)", 0.0, 200.0, 43.0)
            ph_val = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        with col4:
            rain_val = st.number_input("Rainfall (mm)", 0.0, 5000.0, 200.0)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='form-panel'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #2b3a18; margin-bottom: 20px;'>Hierarchical Geo-Spatial Agronomic Intelligence</h3>", unsafe_allow_html=True)
        
        map_col, form_col = st.columns([1, 1.2])
        
        with map_col:
            render_interactive_map()
            
        with form_col:
            hierarchy_data = render_hierarchy_panel()
            
        # --- Live Weather Intelligence Panel (Phase 6Z) ---
        lat_coord = st.session_state.get('map_lat')
        lon_coord = st.session_state.get('map_lng')
        if lat_coord and lon_coord:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div style='background: #f8faf8; border: 1px solid rgba(85, 107, 47, 0.15); border-radius: 12px; padding: 1rem;'>📍 <strong>Selected Coordinates:</strong> {lat_coord:.4f}°N, {lon_coord:.4f}°E</div>", unsafe_allow_html=True)
            
            w_col1, w_col2 = st.columns([1.5, 2])
            with w_col1:
                st.markdown("<br>", unsafe_allow_html=True)
                fetch_weather = st.button("🌍 Fetch Environmental Conditions")
                
            if fetch_weather:
                with st.spinner("Connecting to Open-Meteo real-time sensors..."):
                    from weather.weather_context_engine import get_live_weather_context
                    user_inputs = {
                        "temperature": temp_val,
                        "humidity": hum_val,
                        "rainfall": rain_val,
                        "irrigation_type": hierarchy_data.get("irrigation_type", "rain-fed")
                    }
                    weather_report = get_live_weather_context(lat_coord, lon_coord, user_inputs)
                    st.session_state['weather_context'] = weather_report
                    st.success("✅ Real-time environmental context fetched and cached successfully!")
                    
            if 'weather_context' in st.session_state:
                from frontend.components.weather_cards import render_weather_dashboard
                render_weather_dashboard(st.session_state['weather_context'])
            
        st.markdown("<br>", unsafe_allow_html=True)
        generate = st.button("Discover Strategy ➔")
        st.markdown("</div>", unsafe_allow_html=True)

    if generate:
        with st.spinner("Analyzing spatial hierarchy & soil profile..."):
            payload = {
                "N": n_val, "P": p_val, "K": k_val,
                "temperature": temp_val, "humidity": hum_val,
                "ph": ph_val, "rainfall": rain_val,
                "latitude": lat_coord,
                "longitude": lon_coord,
                "weather_context": st.session_state.get('weather_context'),
                **hierarchy_data
            }
            
            result = predict_soil(payload)
            time.sleep(0.5) 
            
            if result.get("status") == "success":
                preds = result["predictions"]
                
                st.markdown("<br><br><h2 style='color:#2b3a18; text-align:center; margin-bottom:30px;'>AI Strategy Results</h2>", unsafe_allow_html=True)
                
                # --- Square Blocks Grid (Reference Image 2) ---
                crop = preds["crop"].get("prediction", "N/A")
                fert = preds["fertility"].get("prediction", "N/A")
                defi = preds["deficiency"].get("prediction", "N/A")
                
                st.markdown(f"""
                <div class="block-grid">
                    <div class="square-block block-green">
                        <div class="block-icon">🌱</div>
                        <div class="metric-title">Optimal Crop</div>
                        <div class="metric-value">{str(crop).capitalize()}</div>
                    </div>
                    <div class="square-block block-tan">
                        <div class="block-icon">📊</div>
                        <div class="metric-title">Fertility Grade</div>
                        <div class="metric-value">{str(fert).capitalize()}</div>
                    </div>
                    <div class="square-block block-brown">
                        <div class="block-icon">💧</div>
                        <div class="metric-title">Nutrient Status</div>
                        <div class="metric-value" style="font-size:1.8rem;">{str(defi).capitalize()}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # --- Phase 6 Decision Intelligence Layer ---
                dec_support = result.get("decision_support", {})
                if dec_support and "error" not in dec_support:
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # 1. Dual Column layout for core reasoning & narratives
                    col_reasoning, col_actions = st.columns([1, 1])
                    with col_reasoning:
                        # Confidence indicator & Score gauge
                        render_confidence_indicator(dec_support.get("confidence", {}))
                        render_hybrid_intelligence_score_gauge(dec_support.get("hybrid_intelligence_score", {}))
                        
                    with col_actions:
                        # Region Intelligence Panel
                        reg_intel = dec_support.get("region_intelligence", {})
                        if reg_intel:
                            st.markdown(f"#### 🌍 {reg_intel.get('title', 'Region Intelligence')}")
                            if reg_intel.get("environmental_context"):
                                st.info("Context: " + " ".join(reg_intel["environmental_context"]))
                            if reg_intel.get("environmental_risks"):
                                st.warning("Risks: " + " ".join(reg_intel["environmental_risks"]))
                            st.markdown("---")
                            
                        # Agronomic narrative & Prioritized recommendation cards
                        render_agronomic_narrative(dec_support.get("narrative", ""))
                        render_prioritized_recommendations(dec_support.get("prioritized_actions", {}))
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # 2. Alternative Crop Suitability Index
                    render_comparative_crops(dec_support.get("top_k_crops", []), dec_support.get("crops_to_avoid", []))
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # 3. Live Weather Environmental Context Integration (Phase 6Z)
                    w_intel = result.get("weather_intelligence", {})
                    if w_intel and w_intel.get("status") in ["success", "fallback"]:
                        st.markdown("### 🌍 Real-Time Environmental Sensors Report")
                        from frontend.components.weather_cards import render_weather_dashboard
                        render_weather_dashboard(w_intel)
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                    # 4. Connected Agronomic Knowledge Graph & Adaptive Intelligence (Phase 7)
                    a_intel = result.get("adaptive_intelligence", {})
                    if a_intel and "intelligence_score" in a_intel:
                        st.markdown("### 🕸️ Connected Agricultural Adaptive Intelligence")
                        from frontend.components.adaptive_intelligence_cards import render_adaptive_dashboard
                        render_adaptive_dashboard(a_intel)
                        st.markdown("<br>", unsafe_allow_html=True)

                # --- Explainability Panel ---
                st.markdown("### 🔍 Model Explainability & Local SHAP Analysis")
                expls = result.get("explanations", {})
                if "feature_importance" in expls and "error" not in expls["feature_importance"]:
                    plot_feature_importance(expls["feature_importance"])
            else:
                st.error(f"Inference Engine Failed: {result.get('message', 'Unknown error')}")

def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #4d7c0f;'>AGROSPHERE</h2>", unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio("Navigation", ["Precision Services", "Operational Observability"])
        st.markdown("---")
        st.caption("AI Operations Dashboard v5.0")
        return page

def page_diagnostics():
    from frontend.services.api_client import get_system_health, get_system_metrics
    st.markdown("<p class='main-header'>Operational Observability</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>System metrics, latency tracking, and inference analytics.</p>", unsafe_allow_html=True)
    
    st.markdown("### System Health", unsafe_allow_html=True)
    health = get_system_health()
    if health.get("status") == "healthy":
        st.success("🟢 API is ONLINE and HEALTHY")
    else:
        st.error(f"🔴 API OFFLINE: {health.get('message')}")

    st.markdown("### Operational Metrics", unsafe_allow_html=True)
    metrics = get_system_metrics()
    
    if "error" in metrics:
        st.warning("Metrics endpoint currently unreachable.")
        return
        
    st.markdown("<div class='form-panel'>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Requests", metrics.get("total_requests", 0))
    with col2:
        st.metric("Error Rate", f"{metrics.get('error_rate_percent', 0)}%")
    with col3:
        st.metric("Avg Latency", f"{metrics.get('average_latency_ms', 0)} ms")
    with col4:
        st.metric("Cache Hit Ratio", f"{metrics.get('cache_hit_ratio', 0) * 100}%")
        
    st.markdown("---")
    
    st.markdown("#### Inference Analytics (Top Models)")
    preds = metrics.get("top_predictions", {})
    if preds:
        st.bar_chart(preds)
    else:
        st.info("No predictions recorded in this session.")
        
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    page = render_sidebar()
    if page == "Precision Services":
        page_predictive_analysis()
    else:
        page_diagnostics()

if __name__ == "__main__":
    main()
