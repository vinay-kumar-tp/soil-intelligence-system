"""Phase 6 - Premium Decision Support UI Components.

Provides beautiful olive-green and white CSS-styled components to render agronomic 
narratives, confidence tiers, top-K comparative crop metrics, and hybrid scores.
Uses an advanced HTML flattener to prevent standard Markdown code-block glitches.
"""

import streamlit as st

def render_html(html_str: str):
    """Flattens HTML into a single contiguous string to bypass Markdown code-block parsing triggers."""
    flat_html = " ".join([line.strip() for line in html_str.split("\n") if line.strip()])
    st.markdown(flat_html, unsafe_allow_html=True)

def render_confidence_indicator(conf_dict: dict):
    """Renders confidence tier with custom matching alerts."""
    tier = conf_dict.get("tier", "MODERATE CONFIDENCE")
    message = conf_dict.get("message", "")
    
    if "HIGH" in tier:
        bg_color = "#ecfdf5"
        border_color = "#10b981"
        text_color = "#065f46"
        icon = "🟢"
    elif "MODERATE" in tier:
        bg_color = "#fffbeb"
        border_color = "#f59e0b"
        text_color = "#92400e"
        icon = "🟡"
    else:
        bg_color = "#fef2f2"
        border_color = "#ef4444"
        text_color = "#991b1b"
        icon = "🔴"
        
    html_content = f"""
    <div style="background: {bg_color}; border-left: 5px solid {border_color}; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <span style="font-weight: 800; font-size: 0.9rem; text-transform: uppercase; color: {text_color}; display: block; letter-spacing: 1px;">
            {icon} {tier}
        </span>
        <span style="color: #4b5563; font-size: 1rem; font-weight: 500;">{message}</span>
    </div>
    """
    render_html(html_content)

def render_hybrid_intelligence_score_gauge(score_dict: dict):
    """Renders premium gauge with dynamic feedback."""
    score = score_dict.get("score", 70)
    reasons = score_dict.get("reasons", [])
    
    if score >= 80:
        bar_color = "#65a30d"  # vibrant olive/lime green
    elif score >= 60:
        bar_color = "#d4a373"  # sand/tan
    else:
        bar_color = "#b91c1c"  # deep rust red
        
    html_content = f"""
    <div style="background: #f8faf8; border: 1px solid rgba(85, 107, 47, 0.15); border-radius: 16px; padding: 1.5rem; text-align: center; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(85, 107, 47, 0.04);">
        <span style="font-weight: 700; color: #556b2f; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1.5px; display: block; margin-bottom: 0.5rem;">
            Agronomic Intelligence Score
        </span>
        <div style="font-size: 3.5rem; font-weight: 800; color: #2b3a18; line-height: 1;">
            {score}<span style="font-size: 1.5rem; color: #6b7280; font-weight: 500;">/100</span>
        </div>
        <div style="background: #e5e7eb; border-radius: 10px; height: 10px; width: 80%; margin: 1rem auto; overflow: hidden;">
            <div style="background: {bar_color}; width: {score}%; height: 100%; border-radius: 10px;"></div>
        </div>
        <div style="text-align: left; max-width: 90%; margin: 0 auto; border-top: 1px solid #e5e7eb; padding-top: 0.75rem; margin-top: 0.75rem;">
            <span style="font-weight: 700; color: #4b5563; font-size: 0.8rem; text-transform: uppercase; display: block; margin-bottom: 4px;">Score Justification:</span>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.85rem; color: #6b7280;">
                {"".join([f"<li>{r}</li>" for r in reasons])}
            </ul>
        </div>
    </div>
    """
    render_html(html_content)

def render_agronomic_narrative(narrative_text: str):
    """Renders scientific crop-soil narrative block."""
    html_content = f"""
    <div style="background: #ffffff; border: 1px dashed rgba(85, 107, 47, 0.3); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.02);">
        <h4 style="margin-top: 0; color: #2b3a18; font-size: 1.1rem; display: flex; align-items: center; gap: 8px; font-weight: 700;">
            📖 AI Agronomic Narrative
        </h4>
        <p style="color: #4b5563; font-size: 0.98rem; line-height: 1.6; margin: 0; text-align: justify; font-style: italic;">
            " {narrative_text} "
        </p>
    </div>
    """
    render_html(html_content)

def render_prioritized_recommendations(priority_dict: dict):
    """Renders prioritized recommendations with elegant badges."""
    high = priority_dict.get("high", [])
    moderate = priority_dict.get("moderate", [])
    optional = priority_dict.get("optional", [])
    
    html_content = f"""
    <style>
    .priority-card {{
        background: #ffffff;
        border: 1px solid rgba(85, 107, 47, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }}
    .p-section {{ margin-bottom: 1.25rem; }}
    .p-section:last-child {{ margin-bottom: 0; }}
    .p-badge {{
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 800;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.5rem;
    }}
    .badge-high {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }}
    .badge-mod {{ background: #fef3c7; color: #d97706; border: 1px solid #fcd34d; }}
    .badge-opt {{ background: #dcfce7; color: #15803d; border: 1px solid #86efac; }}
    .p-list {{ margin: 0; padding-left: 1.2rem; color: #4b5563; font-size: 0.95rem; line-height: 1.5;}}
    </style>
    <div class="priority-card">
        <h4 style="margin-top: 0; color: #2b3a18; font-size: 1.1rem; margin-bottom: 1rem; font-weight: 700;">
            📋 Agronomic Action Priorities
        </h4>
        <div class="p-section">
            <span class="p-badge badge-high">High Priority (Immediate Correction)</span>
            <ul class="p-list">
                {"".join([f"<li>{item}</li>" for item in high])}
            </ul>
        </div>
        <div class="p-section">
            <span class="p-badge badge-mod">Moderate Priority (Cultivation Support)</span>
            <ul class="p-list">
                {"".join([f"<li>{item}</li>" for item in moderate])}
            </ul>
        </div>
        <div class="p-section">
            <span class="p-badge badge-opt">Optional Optimizations (Long-term gains)</span>
            <ul class="p-list">
                {"".join([f"<li>{item}</li>" for item in optional])}
            </ul>
        </div>
    </div>
    """
    render_html(html_content)

def render_comparative_crops(top_k_crops: list, crops_to_avoid: list = None):
    """Renders a stunning horizontal comparative list of Top 3 compatible crops and a list of crops to avoid."""
    st.markdown("""
    <style>
    .comp-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-bottom: 2rem;
    }
    .comp-card {
        background: #ffffff;
        border: 1px solid rgba(85, 107, 47, 0.12);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px rgba(85, 107, 47, 0.03);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .comp-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(85, 107, 47, 0.08);
    }
    .comp-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(0,0,0,0.05);
        padding-bottom: 8px;
    }
    .comp-name { font-size: 1.2rem; font-weight: 800; color: #2b3a18; }
    .comp-score { font-size: 0.85rem; font-weight: 700; color: #4d7c0f; background: #ecfdf5; border: 1px solid #a7f3d0; padding: 3px 10px; border-radius: 20px;}
    .comp-desc { font-size: 0.85rem; color: #64748b; margin-bottom: 12px; line-height: 1.45; font-style: italic;}
    .comp-section-title { font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase; margin-top: 10px; margin-bottom: 4px; display: block;}
    .comp-bullets { margin: 0; padding-left: 1.2rem; font-size: 0.85rem; color: #334155; line-height: 1.45;}
    
    /* Avoid Section Styling */
    .avoid-title {
        color: #991b1b !important;
        font-size: 1.3rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        font-weight: 800 !important;
        border-left: 4px solid #ef4444;
        padding-left: 10px;
    }
    .avoid-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-bottom: 1.5rem;
    }
    .avoid-card {
        background: #fff5f5;
        border: 1px solid #fee2e2;
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 4px 10px rgba(153, 27, 27, 0.02);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s;
    }
    .avoid-card:hover {
        transform: translateY(-3px);
    }
    .avoid-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .avoid-name { font-size: 1.15rem; font-weight: 800; color: #991b1b; }
    .avoid-score { font-size: 0.8rem; font-weight: 700; color: #b91c1c; background: #fee2e2; padding: 2px 8px; border-radius: 12px; border: 1px solid #fca5a5;}
    .avoid-reason { font-size: 0.85rem; color: #7f1d1d; line-height: 1.45; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🌾 Alternative Crop Suitability Index")
    
    card_htmls = []
    for crop in top_k_crops:
        adv_html = "".join([f"<li>{a}</li>" for a in crop.get("advantages", [])])
        risk_html = "".join([f"<li>{r}</li>" for r in crop.get("risks", [])])
        limit_html = "".join([f"<li>{l}</li>" for l in crop.get("limiting_factors", [])])
        
        card_htmls.append(f"""
        <div class="comp-card">
            <div>
                <div class="comp-header">
                    <span class="comp-name">{crop['crop']}</span>
                    <span class="comp-score">{crop['suitability_score']}% Match</span>
                </div>
                <div class="comp-desc">"{crop['description']}"</div>
                <span class="comp-section-title" style="color: #166534;">✚ Climatic Advantages:</span>
                <ul class="comp-bullets">{adv_html}</ul>
                <span class="comp-section-title" style="color: #991b1b;">⚠ Agronomic Risks:</span>
                <ul class="comp-bullets">{risk_html}</ul>
            </div>
            <div style="margin-top: 12px; border-top: 1px dashed rgba(0,0,0,0.05); padding-top: 8px;">
                <span class="comp-section-title" style="color: #d97706;">✖ Limiting Factors:</span>
                <ul class="comp-bullets">{limit_html}</ul>
            </div>
        </div>
        """)
        
    html_container = f"""
    <div class="comp-container">
        {"".join(card_htmls)}
    </div>
    """
    render_html(html_container)
    
    if crops_to_avoid:
        st.markdown("<h4 class='avoid-title'>⚠️ Crops Highly Recommended to Avoid</h4>", unsafe_allow_html=True)
        avoid_cards = []
        for item in crops_to_avoid:
            avoid_cards.append(f"""
            <div class="avoid-card">
                <div>
                    <div class="avoid-header">
                        <span class="avoid-name">❌ {item['crop']}</span>
                        <span class="avoid-score">{item['suitability_score']}% Match</span>
                    </div>
                    <div class="avoid-reason"><strong>Limitation:</strong> {item['reason']}</div>
                </div>
            </div>
            """)
            
        html_avoid = f"""
        <div class="avoid-container">
            {"".join(avoid_cards)}
        </div>
        """
        render_html(html_avoid)
