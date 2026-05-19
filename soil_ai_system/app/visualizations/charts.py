import streamlit as st
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_confidence_gauge(confidence: float, title: str):
    """Render a confidence gauge."""
    if PLOTLY_AVAILABLE:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': title, 'font': {'size': 14, 'color': '#f8fafc'}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': "#34d399"},
                'bgcolor': "#1e293b",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 50], 'color': "#ef4444"},
                    {'range': [50, 80], 'color': "#fbbf24"},
                    {'range': [80, 100], 'color': "#10b981"}],
            }
        ))
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': '#f8fafc'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(title)
        st.progress(min(float(confidence), 1.0))
        st.write(f"{confidence*100:.1f}%")


def plot_feature_importance(shap_data: dict):
    """Render SHAP feature importance as a bar chart."""
    contributions = shap_data.get("feature_contributions", {})
    if not contributions:
        st.info("No feature importance data available.")
        return
        
    df = pd.DataFrame(list(contributions.items()), columns=["Feature", "Impact"])
    # Sort for bottom-up bar chart display
    df = df.sort_values(by="Impact", ascending=True).tail(10)
    
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            df, x="Impact", y="Feature", orientation='h',
            title="Local SHAP Feature Importance",
            labels={"Impact": "Mean Absolute SHAP Value"},
            color="Impact",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(data=df, x="Feature", y="Impact")
