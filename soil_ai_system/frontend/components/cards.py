import streamlit as st

def render_recommendation_card(recommendations: dict):
    """Render enterprise-grade recommendation cards."""
    crop_rationale = recommendations.get("crop_rationale", [])
    health_actions = recommendations.get("soil_health_actions", [])
    fert_rec = recommendations.get("fertilizer_recommendation", "")
    
    # Custom HTML for Dark Mode Enterprise styling
    st.markdown("""
    <style>
    .rec-panel {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .rec-title { margin-top: 0; margin-bottom: 0.5rem; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;}
    .rec-blue { border-left: 4px solid #3b82f6; }
    .rec-red { border-left: 4px solid #ef4444; }
    .rec-green { border-left: 4px solid #10b981; }
    .rec-list { margin: 0; padding-left: 1.2rem; color: #cbd5e1; font-size: 0.95rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rec-panel rec-blue">
        <h4 class="rec-title">🌱 Output Rationale</h4>
        <ul class="rec-list">
    """ + "".join([f"<li>{r}</li>" for r in crop_rationale]) + """
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if health_actions:
        st.markdown("""
        <div class="rec-panel rec-red">
            <h4 class="rec-title">⚠️ Required Soil Health Actions</h4>
            <ul class="rec-list">
        """ + "".join([f"<li>{a}</li>" for a in health_actions]) + """
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="rec-panel rec-green">
        <h4 class="rec-title">💧 Fertilizer Optimization Strategy</h4>
        <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">{fert_rec}</p>
    </div>
    """, unsafe_allow_html=True)


def render_contrastive_panel(contrastive_data: dict):
    """Render the contrastive explanation panel."""
    exp = contrastive_data.get("explanation", "")
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid #334155; padding: 1.25rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="margin-top:0; color: #e2e8f0; font-size: 1.1rem;">🧠 Inference Confidence Engine</h4>
        <p style="margin-bottom:0; color: #94a3b8; font-size: 0.95rem;">
            {exp}
        </p>
    </div>
    """, unsafe_allow_html=True)
