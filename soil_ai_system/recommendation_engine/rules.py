from config import N_LOW, P_LOW, K_LOW, MOISTURE_LOW, MOISTURE_HIGH, PH_MIN, PH_MAX


def fertilizer_recommendation(N, P, K, ph, state=None, crop=None, season=None):
    """Generate fertilizer guidance based on soil chemistry and context.

    Args:
        N (float): Nitrogen value.
        P (float): Phosphorus value.
        K (float): Potassium value.
        ph (float): Soil pH value.
        state (str | None): Optional region context.
        crop (str | None): Predicted crop label.
        season (str | None): Season label.

    Returns:
        list[str]: Fertilizer recommendation statements.
    """
    recs = []

    if N < N_LOW:
        dose = "50 kg/acre" if season == "kharif" else "35 kg/acre"
        recs.append(f"Add Urea (Nitrogen) - {dose} recommended")
    if P < P_LOW:
        recs.append("Add Single Super Phosphate (SSP) - 30 kg/acre recommended")
    if K < K_LOW:
        recs.append("Add Muriate of Potash (MOP) - 25 kg/acre recommended")
    if ph < PH_MIN:
        recs.append("Soil acidic - Apply agricultural lime (calcite) 200 kg/acre")
    elif ph > PH_MAX:
        recs.append("Soil alkaline - Apply sulfur 30 kg/acre to reduce pH")

    if state == "Tamil Nadu" and crop == "Rice" and N < N_LOW:
        recs.append("TN-specific: Use neem-coated urea for paddy - reduces leaching")

    if not recs:
        recs.append("Soil is nutrient-balanced - no immediate fertilizer action needed")
    return recs


def irrigation_recommendation(moisture, temperature, humidity, crop=None):
    """Generate irrigation guidance based on moisture and climate.

    Args:
        moisture (float): Soil moisture percentage.
        temperature (float): Temperature value.
        humidity (float): Humidity percentage.
        crop (str | None): Predicted crop label.

    Returns:
        str: Irrigation recommendation message.
    """
    if moisture < MOISTURE_LOW:
        urgency = "URGENT" if moisture < 15 else "Moderate"
        return f"[{urgency}] Irrigation needed - moisture at {moisture}%. Suggested: drip irrigation."
    if moisture > MOISTURE_HIGH:
        return f"WARNING: Soil overwatered ({moisture}%). Pause irrigation. Risk of root rot."
    return f"Optimal moisture ({moisture}%). No irrigation needed currently."


def seasonal_advice(crop, season, state=None):
    """Return season-specific advice for a crop.

    Args:
        crop (str): Crop label.
        season (str): Season label.
        state (str | None): Optional region context.

    Returns:
        str: Seasonal advice message.
    """
    calendar = {
        "Rice": {
            "kharif": "Sow June-July. Flood irrigation.",
            "rabi": "Rabi rice possible in TN - Pishanam/Samba varieties.",
            "summer": "Avoid summer rice - high water demand.",
        },
        "Wheat": {
            "kharif": "Wheat is a Rabi crop - do not sow in Kharif.",
            "rabi": "Sow Nov-Dec. Ideal for North India.",
            "summer": "Not suitable.",
        },
        "Groundnut": {
            "kharif": "Best season. Sow June-July in TN.",
            "rabi": "Second-best. Oct-Nov sowing.",
            "summer": "Possible with irrigation in Feb-Mar.",
        },
        "Sugarcane": {
            "kharif": "Ratoon crop management.",
            "rabi": "Plant cane Feb-Mar for optimal sucrose.",
            "summer": "High water needs - ensure irrigation.",
        },
        "Banana": {
            "kharif": "Transplant June-July. Common in TN.",
            "rabi": "Delayed growth - protect from cold.",
            "summer": "Risk of drought stress - drip irrigation essential.",
        },
    }
    advice = calendar.get(crop, {}).get(
        season, f"{crop} - consult local KVK for {season} season guidance."
    )
    return advice


def crop_action_guide(crop, state=None):
    """Return crop-specific agronomy guidance.

    Args:
        crop (str): Crop label.
        state (str | None): Optional region context.

    Returns:
        str: Crop guidance message.
    """
    guides = {
        "Rice": "Maintain 5-10 cm standing water. Ideal temp 22-30C. Watch for blast disease.",
        "Wheat": "Well-drained loamy soil. Sow Nov-Dec for Rabi. Ensure 6 irrigations.",
        "Sugarcane": "Deep black soil. High K for juice quality. Ratoon management key.",
        "Groundnut": "Sandy loam. Major crop in TN districts - Vellore, Tirunelveli.",
        "Cotton": "Deep black soil. Bt cotton common. Monitor bollworm carefully.",
        "Banana": "High moisture + warm climate. Prominent in Thanjavur, Trichy, TN.",
        "Coconut": "Sandy loam + coastal zones. Palakkad, Coimbatore districts. Salt tolerant.",
        "Maize": "Versatile crop. Kharif + Rabi. Well-drained loamy soil.",
        "Jowar": "Drought-tolerant. Rabi crop. Good for semi-arid TN zones.",
        "Ragi": "Rainfed crop. Excellent for hilly TN regions - Nilgiris, Salem.",
    }
    return guides.get(crop, "Consult local Krishi Vigyan Kendra (KVK) for guidance.")


def full_recommendation(
    N,
    P,
    K,
    ph,
    moisture,
    temperature,
    humidity,
    predicted_crop,
    season=None,
    state=None,
):
    """Aggregate fertilizer, irrigation, seasonal, and crop guidance.

    Args:
        N (float): Nitrogen value.
        P (float): Phosphorus value.
        K (float): Potassium value.
        ph (float): Soil pH value.
        moisture (float): Soil moisture percentage.
        temperature (float): Temperature value.
        humidity (float): Humidity percentage.
        predicted_crop (str): Predicted crop label.
        season (str | None): Season label.
        state (str | None): Optional region context.

    Returns:
        dict: Combined recommendation outputs.
    """
    return {
        "fertilizer": fertilizer_recommendation(N, P, K, ph, state, predicted_crop, season),
        "irrigation": irrigation_recommendation(moisture, temperature, humidity, predicted_crop),
        "seasonal": seasonal_advice(predicted_crop, season or "kharif", state),
        "crop_guide": crop_action_guide(predicted_crop, state),
    }
