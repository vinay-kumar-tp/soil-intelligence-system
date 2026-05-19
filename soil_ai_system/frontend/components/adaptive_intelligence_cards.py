"""Phase 7 - Connected Agricultural Adaptive Intelligence UI Components.

Renders premium adaptive reasoning dashboards, Agronomic Intelligence Scores,
Session Memory trends, Knowledge Graph edges, and Long-Form Reports.
"""

import streamlit as st

def render_adaptive_html(html_str: str):
    """Flattens and renders HTML without standard Markdown block triggers."""
    flat_html = " ".join([line.strip() for line in html_str.split("\n") if line.strip()])
    st.markdown(flat_html, unsafe_allow_html=True)

def render_adaptive_dashboard(adaptive_intel: dict):
    """Renders the comprehensive Phase 7 Adaptive Intelligence Report."""
    if not adaptive_intel or adaptive_intel.get("status") == "error":
        return
        
    score = adaptive_intel.get("intelligence_score", 85)
    soil_health = adaptive_intel.get("soil_health_rating", 80)
    kg_score = adaptive_intel.get("kg_score", 80)
    
    kg_details = adaptive_intel.get("kg_match_details", {})
    matching = kg_details.get("matching", [])
    conflicting = kg_details.get("conflicting", [])
    sust_bonus = kg_details.get("sustainability_bonus", 0)
    
    drift = adaptive_intel.get("session_drift", {})
    recurring = adaptive_intel.get("recurring_deficiencies", [])
    notices = adaptive_intel.get("adaptive_weight_notices", [])
    rebuild = adaptive_intel.get("soil_rebuild_urgency", "standard")
    long_form = adaptive_intel.get("long_form_report", [])
    
    # Dynamic styling for Intelligence Score
    if score >= 80:
        score_color = "#4d7c0f"  # vibrant agricultural green
        score_bg = "#f0fdf4"
        score_border = "#bbf7d0"
    elif score >= 50:
        score_color = "#b45309"  # warm orange
        score_bg = "#fffbeb"
        score_border = "#fef3c7"
    else:
        score_color = "#b91c1c"  # warnings red
        score_bg = "#fef2f2"
        score_border = "#fee2e2"
        
    # Matching elements HTML
    match_list_html = "".join([
        f'<div style="font-size: 0.9rem; color: #1e3a1e; font-weight: 600; padding: 4px 0; display: flex; align-items: center;"><span style="color: #22c55e; margin-right: 8px;">✔</span> {m}</div>'
        for m in matching
    ]) if matching else '<div style="font-size: 0.9rem; color: #64748b; font-style: italic;">No matching graph relationships resolved.</div>'
    
    conflict_list_html = "".join([
        f'<div style="font-size: 0.9rem; color: #7f1d1d; font-weight: 600; padding: 4px 0; display: flex; align-items: center;"><span style="color: #ef4444; margin-right: 8px;">✘</span> {c}</div>'
        for c in conflicting
    ]) if conflicting else '<div style="font-size: 0.9rem; color: #64748b; font-style: italic;">No conflicting graph constraints resolved.</div>'

    # Recurring deficiencies HTML
    recurring_html = "".join([
        f'<span style="background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 700; margin-right: 6px;">{r}</span>'
        for r in recurring
    ]) if recurring else '<span style="color: #64748b; font-size: 0.85rem; font-style: italic;">None spotted in this session</span>'

    # Session Notices HTML
    notices_html = "".join([
        f'<div style="background: #fffbeb; border-left: 4px solid #f59e0b; color: #78350f; padding: 8px 12px; margin-bottom: 6px; border-radius: 4px; font-size: 0.85rem; font-weight: 600;">⚠️ {n}</div>'
        for n in notices
    ])

    # Long-Form Report paragraphs
    paragraphs_html = "".join([
        f'<p style="font-size: 0.95rem; line-height: 1.6; color: #1e293b; font-weight: 500; margin-bottom: 12px;">{p}</p>'
        for p in long_form
    ])

    html_content = f"""
    <div style="background: #ffffff; border: 1px solid rgba(85, 107, 47, 0.15); border-radius: 24px; padding: 2.2rem; margin-top: 2rem; box-shadow: 0 10px 30px rgba(85, 107, 47, 0.06);">
        <!-- Header -->
        <div style="border-bottom: 1px solid rgba(85, 107, 47, 0.1); padding-bottom: 1rem; margin-bottom: 1.8rem; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 800; font-size: 1.3rem; color: #2b3a18; letter-spacing: 0.5px;">🧠 PHASE 7 - CONNECTED AGRICULTURAL INTELLIGENCE</span>
            <span style="background: rgba(77, 124, 15, 0.1); color: #4d7c0f; padding: 4px 12px; border-radius: 50px; font-size: 0.8rem; font-weight: 800;">ADAPTIVE MODE ACTIVE</span>
        </div>
        
        <!-- Score Matrix Dashboard -->
        <div style="display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem;">
            <!-- Main Score -->
            <div style="background: {score_bg}; border: 1px solid {score_border}; border-radius: 16px; padding: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="display: block; font-size: 0.85rem; font-weight: 800; color: {score_color}; text-transform: uppercase; letter-spacing: 1px;">Agronomic Intelligence</span>
                    <span style="font-size: 1rem; font-weight: 700; color: #334155; margin-top: 4px; display: block;">Combined Graph Score</span>
                </div>
                <div style="font-size: 3.2rem; font-weight: 950; color: {score_color}; line-height: 1;">
                    {score}<span style="font-size: 1.5rem; font-weight: 800;">/100</span>
                </div>
            </div>
            
            <!-- Soil Health -->
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="display: block; font-size: 0.85rem; font-weight: 800; color: #475569; text-transform: uppercase; letter-spacing: 1px;">Soil Health</span>
                    <span style="font-size: 1rem; font-weight: 700; color: #334155; margin-top: 4px; display: block;">Chemical Rating</span>
                </div>
                <div style="font-size: 2.5rem; font-weight: 900; color: #334155;">
                    {soil_health}<span style="font-size: 1.2rem; font-weight: 700;">%</span>
                </div>
            </div>
            
            <!-- Graph Compatibility -->
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="display: block; font-size: 0.85rem; font-weight: 800; color: #475569; text-transform: uppercase; letter-spacing: 1px;">Graph Compatibility</span>
                    <span style="font-size: 1rem; font-weight: 700; color: #334155; margin-top: 4px; display: block;">Relation Matcher</span>
                </div>
                <div style="font-size: 2.5rem; font-weight: 900; color: #334155;">
                    {kg_score}<span style="font-size: 1.2rem; font-weight: 700;">%</span>
                </div>
            </div>
        </div>
        
        <!-- Knowledge Graph Relations & Session Memory -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem;">
            <!-- Knowledge Graph Card -->
            <div style="background: #fafaf9; border: 1px solid #e7e5e4; border-radius: 14px; padding: 1.2rem;">
                <span style="display: block; font-size: 0.9rem; font-weight: 800; color: #44403c; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #e7e5e4; padding-bottom: 6px; margin-bottom: 8px;">🕸️ Agronomic Knowledge Graph Traversal</span>
                <div style="margin-bottom: 10px;">
                    <span style="display: block; font-size: 0.8rem; font-weight: 700; color: #78716c; text-transform: uppercase; margin-bottom: 2px;">Reinforcing Edges Match</span>
                    {match_list_html}
                </div>
                <div>
                    <span style="display: block; font-size: 0.8rem; font-weight: 700; color: #78716c; text-transform: uppercase; margin-bottom: 2px;">Conflicting Constraints</span>
                    {conflict_list_html}
                </div>
            </div>
            
            <!-- Session Memory Card -->
            <div style="background: #fafaf9; border: 1px solid #e7e5e4; border-radius: 14px; padding: 1.2rem;">
                <span style="display: block; font-size: 0.9rem; font-weight: 800; color: #44403c; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #e7e5e4; padding-bottom: 6px; margin-bottom: 8px;">🧠 Session Memory & Environmental Drift</span>
                <div style="margin-bottom: 10px;">
                    <span style="display: block; font-size: 0.8rem; font-weight: 700; color: #78716c; text-transform: uppercase; margin-bottom: 4px;">Recurring Soil Deficiencies Detected</span>
                    <div>{recurring_html}</div>
                </div>
                <div style="margin-bottom: 10px; display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; color: #44403c;">
                    <span>Drought Trend: <strong style="color: {'#ef4444' if drift.get('drought_trend') else '#22c55e'};">{ 'Active' if drift.get('drought_trend') else 'Negative' }</strong></span>
                    <span>Rainfall Trend: <strong style="color: #3b82f6;">{ str(drift.get('rainfall_trend', 'stable')).capitalize() }</strong></span>
                    <span>Temp Drift: <strong>{ drift.get('temp_drift', 0.0) }°C</strong></span>
                </div>
            </div>
        </div>
        
        <!-- Adaptive Adjustments notices -->
        {f'''
        <div style="margin-bottom: 1.8rem;">
            {notices_html}
        </div>
        ''' if notices else ''}
        
        <!-- Premium Long-Form Agricultural Report -->
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.8rem;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">
                <span style="font-size: 1.5rem; margin-right: 10px;">📋</span>
                <span style="font-weight: 800; font-size: 1rem; color: #1e293b; letter-spacing: 0.5px;">PREMIUM AI AGRONOMIC NARRATIVE & DECISION REPORT</span>
            </div>
            {paragraphs_html}
        </div>
    </div>
    """
    render_adaptive_html(html_content)
