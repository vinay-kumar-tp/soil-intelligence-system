"""Phase 6Z - Premium Real-time Weather & Environmental Intelligence UI Components.

Renders beautiful HTML widgets for live weather conditions, short-term forecasts,
environmental alignment scores, and risk analysis indicators.
"""

import streamlit as st

def render_weather_html(html_str: str):
    """Flattens and renders HTML without standard Markdown block triggers."""
    flat_html = " ".join([line.strip() for line in html_str.split("\n") if line.strip()])
    st.markdown(flat_html, unsafe_allow_html=True)

def render_weather_dashboard(weather_report: dict):
    """Renders a gorgeous dashboard of live weather and environmental alignment."""
    if not weather_report or weather_report.get("status") == "error":
        return
        
    weather = weather_report.get("weather", {})
    compatibility = weather_report.get("compatibility", {})
    
    # 1. Weather Info
    temp = weather.get("temperature", 25.0)
    hum = weather.get("humidity", 50.0)
    rain = weather.get("rainfall", 0.0)
    wind = weather.get("wind_speed", 0.0)
    condition = weather.get("weather_condition", "Clear sky")
    forecast = weather.get("forecast", {})
    precip_7d = forecast.get("predicted_precipitation_sum_7d", 0.0)
    
    # 2. Compatibility & Alignment
    alignment_score = compatibility.get("alignment_score", 100)
    realism_tier = compatibility.get("realism_confidence", "HIGH")
    warnings = compatibility.get("consistency_warnings", [])
    narratives = compatibility.get("narratives", [])
    badges = compatibility.get("climate_badges", [])
    
    # Colors based on alignment
    if alignment_score >= 80:
        align_color = "#65a30d"  # vibrant green
        align_bg = "#ecfdf5"
    elif alignment_score >= 50:
        align_color = "#d4a373"  # tan/sand
        align_bg = "#fffbeb"
    else:
        align_color = "#b91c1c"  # deep warning red
        align_bg = "#fef2f2"
        
    # Badges HTML
    badges_html = "".join([
        f'<span style="background: rgba(85, 107, 47, 0.1); color: #556b2f; padding: 0.2rem 0.6rem; border-radius: 50px; font-size: 0.8rem; font-weight: 700; margin-right: 0.5rem; text-transform: uppercase;">{b}</span>'
        for b in badges
    ])
    
    # Risks Info
    risks = compatibility.get("risks", {})
    drought = risks.get("drought_risk", "Low")
    flood = risks.get("flood_sensitivity", "Low")
    heat = risks.get("heat_stress", "Low")
    water = risks.get("water_stress", "Low")
    instability = risks.get("instability_warnings", [])
    
    # Risk badges generator
    def risk_badge(level: str) -> str:
        color = "#10b981" if level == "Low" else ("#f59e0b" if level == "Medium" else "#ef4444")
        return f'<span style="color: {color}; font-weight: 800;">{level}</span>'

    html_content = f"""
    <div style="background: #ffffff; border: 1px solid rgba(85, 107, 47, 0.15); border-radius: 20px; padding: 1.8rem; margin-top: 1.5rem; box-shadow: 0 4px 20px rgba(85, 107, 47, 0.04);">
        <!-- Title & Badges -->
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(85, 107, 47, 0.1); padding-bottom: 0.8rem; margin-bottom: 1.2rem;">
            <span style="font-weight: 800; font-size: 1.1rem; color: #2b3a18; letter-spacing: 0.5px;">🌍 LIVE ENVIRONMENTAL AGRICULTURE REPORT</span>
            <div>{badges_html}</div>
        </div>
        
        <!-- Live Weather Metrics Grid -->
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
            <div style="background: #f8fafc; border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid #e2e8f0;">
                <span style="display: block; font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Temperature</span>
                <span style="font-size: 1.6rem; font-weight: 800; color: #1e293b;">{temp:.1f}°C</span>
                <span style="display: block; font-size: 0.75rem; font-weight: 600; color: #556b2f; margin-top: 2px;">{condition}</span>
            </div>
            
            <div style="background: #f8fafc; border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid #e2e8f0;">
                <span style="display: block; font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Humidity</span>
                <span style="font-size: 1.6rem; font-weight: 800; color: #1e293b;">{hum:.1f}%</span>
                <span style="display: block; font-size: 0.75rem; font-weight: 600; color: #64748b; margin-top: 2px;">Atmospheric</span>
            </div>
            
            <div style="background: #f8fafc; border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid #e2e8f0;">
                <span style="display: block; font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Wind Speed</span>
                <span style="font-size: 1.6rem; font-weight: 800; color: #1e293b;">{wind:.1f} <span style="font-size: 0.9rem;">km/h</span></span>
                <span style="display: block; font-size: 0.75rem; font-weight: 600; color: #64748b; margin-top: 2px;">Surface velocity</span>
            </div>
            
            <div style="background: #f8fafc; border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid #e2e8f0;">
                <span style="display: block; font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase;">7D Forecast Rain</span>
                <span style="font-size: 1.6rem; font-weight: 800; color: #1e293b;">{precip_7d:.1f} <span style="font-size: 0.9rem;">mm</span></span>
                <span style="display: block; font-size: 0.75rem; font-weight: 600; color: #64748b; margin-top: 2px;">Short-term sum</span>
            </div>
        </div>
        
        <!-- Reality Validation Section -->
        <div style="background: {align_bg}; border-left: 5px solid {align_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1.2rem; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="display: block; font-weight: 800; font-size: 0.85rem; color: {align_color}; text-transform: uppercase; letter-spacing: 0.5px;">Sensor Reality Alignment Scorer</span>
                <span style="font-size: 0.95rem; font-weight: 600; color: #334155;">Confidence Status: <strong style="color: {align_color};">{realism_tier}</strong></span>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 2.2rem; font-weight: 900; color: {align_color}; line-height: 1;">{alignment_score}<span style="font-size: 1.2rem; font-weight: 700;">%</span></span>
            </div>
        </div>
        
        <!-- Warnings / Compatibility Narratives -->
        {f'''
        <div style="margin-bottom: 1.2rem;">
            <span style="display: block; font-weight: 800; font-size: 0.85rem; color: #b91c1c; text-transform: uppercase; margin-bottom: 4px;">⚠️ Sensor Consistency Warnings</span>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem; color: #7f1d1d; font-weight: 500;">
                {"".join([f'<li>{w}</li>' for w in warnings])}
            </ul>
        </div>
        ''' if warnings else ''}
        
        <div style="margin-bottom: 1.2rem;">
            <span style="display: block; font-weight: 800; font-size: 0.85rem; color: #556b2f; text-transform: uppercase; margin-bottom: 4px;">🌱 Environmental Compatibility Signals</span>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem; color: #334155; font-weight: 500;">
                {"".join([f'<li>{n}</li>' for n in narratives])}
            </ul>
        </div>

        <!-- Risk Matrix Table -->
        <div style="border-top: 1px solid rgba(85, 107, 47, 0.1); padding-top: 1rem;">
            <span style="display: block; font-weight: 800; font-size: 0.85rem; color: #2b3a18; text-transform: uppercase; margin-bottom: 8px;">🔥 Environmental Risk Matrix</span>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; font-size: 0.9rem;">
                <div>Drought Risk: {risk_badge(drought)}</div>
                <div>Flood Sensitivity: {risk_badge(flood)}</div>
                <div>Heat Stress: {risk_badge(heat)}</div>
                <div>Water Stress: {risk_badge(water)}</div>
            </div>
            
            {f'''
            <div style="background: #fffbeb; border: 1px solid #fef3c7; border-radius: 8px; padding: 0.8rem; margin-top: 0.8rem;">
                <span style="color: #b45309; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; display: block; margin-bottom: 2px;">⚠️ Dynamic Instability Alerts</span>
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.85rem; color: #78350f; font-weight: 500;">
                    {"".join([f'<li>{x}</li>' for x in instability])}
                </ul>
            </div>
            ''' if instability else ''}
        </div>
    </div>
    """
    render_weather_html(html_content)
