import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.append("..")

st.set_page_config(
    page_title="Soil Intelligence System",
    page_icon="SOIL",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stMetric { background:#f0f7f0; border-radius:10px; padding:8px; }
.health-gauge { text-align:center; font-size:48px; font-weight:bold; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("AI-Powered Spatial Soil Intelligence System")
st.caption("Explainable AI for Crop Recommendation and Soil Health - v2.0")

st.sidebar.header("Enter Soil Parameters")
N = st.sidebar.slider("Nitrogen (N) kg/ha", 0, 200, 50)
P = st.sidebar.slider("Phosphorus (P) kg/ha", 0, 200, 40)
K = st.sidebar.slider("Potassium (K) kg/ha", 0, 200, 40)
ph = st.sidebar.slider("pH Level", 0.0, 14.0, 6.5, 0.1)
moisture = st.sidebar.slider("Moisture percent", 0, 100, 40)
temperature = st.sidebar.slider("Temperature C", 0, 50, 28)
humidity = st.sidebar.slider("Humidity percent", 0, 100, 65)
rainfall = st.sidebar.slider("Annual Rainfall mm", 0, 3000, 800)
ec = st.sidebar.number_input("Electrical Conductivity", 0.0, 5.0, 0.5)
organic_carbon = st.sidebar.number_input("Organic Carbon percent", 0.0, 5.0, 0.8)
state = st.sidebar.selectbox(
    "State/Region",
    ["Tamil Nadu", "Karnataka", "Andhra Pradesh", "Kerala", "Maharashtra", "Punjab"],
)
season = st.sidebar.selectbox("Season", ["kharif", "rabi", "summer"])
predict_btn = st.sidebar.button("Analyze Soil and Predict", type="primary")

if predict_btn:
    with st.spinner("Running AI analysis..."):
        from inference.predict import run_full_inference

        result = run_full_inference(
            {
                "N": N,
                "P": P,
                "K": K,
                "ph": ph,
                "moisture": moisture,
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "ec": ec,
                "organic_carbon": organic_carbon,
                "state": state,
                "season": season,
            }
        )

    st.subheader("Soil Health Score")
    score = result["soil_health_score"]
    color = "#2ecc71" if score >= 70 else "#f39c12" if score >= 40 else "#e74c3c"
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            title={"text": "Soil Health Score / 100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#fadbd8"},
                    {"range": [40, 70], "color": "#fdebd0"},
                    {"range": [70, 100], "color": "#d5f5e3"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "value": 70},
            },
        )
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recommended Crop", result["crop"])
        st.progress(result["confidence_crop"])
        st.caption(f"Confidence: {result['confidence_crop'] * 100:.1f}%")
    with col2:
        st.metric("Fertility Grade", result["fertility_grade"])
        st.progress(result["confidence_fertility"])
        st.caption(f"Confidence: {result['confidence_fertility'] * 100:.1f}%")
    with col3:
        st.metric("Nutrient Status", result["nutrient_status"])

    st.divider()

    st.subheader("Why This Prediction (SHAP)")
    shap_top = result.get("shap_top_features", [])
    if shap_top:
        shap_df = pd.DataFrame(shap_top)
        fig_shap = px.bar(
            shap_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="RdYlGn",
            title="Top Features Influencing Crop Prediction",
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    st.subheader("Why Not Another Crop")
    contrastive = result.get("contrastive_explanation", {})
    if contrastive:
        st.info(
            f"{contrastive.get('predicted_crop')} was chosen over "
            f"{contrastive.get('runner_up_crop')}.\n\n"
            f"{contrastive.get('why_not_runner_up')}"
        )

    st.divider()

    st.subheader("Actionable Recommendations")
    r1, r2 = st.columns(2)
    with r1:
        st.write("Fertilizer Actions:")
        for rec in result.get("fertilizer_recommendations", []):
            st.write(f"- {rec}")
        st.write(f"Seasonal Advice ({season.title()}):")
        st.success(result.get("seasonal_advice", ""))
    with r2:
        st.write("Irrigation Status:")
        st.info(result.get("irrigation_suggestion", ""))
        st.write("Crop Guide:")
        st.success(result.get("crop_action_guide", ""))

    st.subheader("Soil Profile Radar Chart")
    cats = ["Nitrogen", "Phosphorus", "Potassium", "pH (x10)", "Moisture", "Humidity"]
    vals = [N / 2, P / 2, K / 2, ph * 10, moisture, humidity]
    fig_radar = go.Figure(
        go.Scatterpolar(r=vals, theta=cats, fill="toself", line_color="green", name="Your Soil")
    )
    ideal_rice = [55, 45, 45, 65, 50, 65]
    fig_radar.add_trace(
        go.Scatterpolar(
            r=ideal_rice, theta=cats, fill="toself", line_color="blue", name="Ideal (Rice)", opacity=0.3
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    import os

    if os.path.exists("reports/shap_outputs/shap_waterfall.png"):
        st.subheader("SHAP Waterfall Plot")
        st.image("reports/shap_outputs/shap_waterfall.png", use_column_width=True)
