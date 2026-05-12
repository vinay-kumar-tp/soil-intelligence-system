import pandas as pd
from preprocessing.validator import validate_input
from preprocessing.feature_engineer import create_soil_quality_index


def test_valid_input():
    """Validate that a compliant input passes validation.

    Args:
        None

    Returns:
        None
    """
    inp = {
        "N": 50,
        "P": 40,
        "K": 40,
        "ph": 6.5,
        "moisture": 40,
        "humidity": 65,
        "rainfall": 800,
        "temperature": 28,
        "ec": 0.5,
        "organic_carbon": 0.8,
    }
    result = validate_input(inp)
    assert result["valid"] is True


def test_invalid_ph():
    """Validate that out-of-range pH fails validation.

    Args:
        None

    Returns:
        None
    """
    inp = {
        "N": 50,
        "P": 40,
        "K": 40,
        "ph": 15,
        "moisture": 40,
        "humidity": 65,
        "rainfall": 800,
        "temperature": 28,
    }
    result = validate_input(inp)
    assert result["valid"] is False
    assert any("pH" in err for err in result["errors"])


def test_soil_quality_index():
    """Ensure soil_quality_index feature is computed.

    Args:
        None

    Returns:
        None
    """
    df = pd.DataFrame([
        {"N": 50, "P": 40, "K": 40, "ph": 6.5, "organic_carbon": 0.8}
    ])
    df = create_soil_quality_index(df)
    assert "soil_quality_index" in df.columns
    assert df["soil_quality_index"].iloc[0] > 0
