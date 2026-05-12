import pandas as pd

from preprocessing.cleaner import remove_duplicates
from preprocessing.encoder import encode_labels
from preprocessing.feature_engineer import create_soil_quality_index
from preprocessing.validator import validate_input


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
    assert any("ph" in err.lower() for err in result["errors"])


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


def test_remove_duplicates():
    """Ensure duplicate removal logs and reduces rows.

    Args:
        None

    Returns:
        None
    """
    df = pd.DataFrame([{"N": 10}, {"N": 10}, {"N": 20}])
    cleaned = remove_duplicates(df)
    assert len(cleaned) == 2


def test_label_encoding_unknown():
    """Validate unknown labels map to the unknown value.

    Args:
        None

    Returns:
        None
    """
    df_train = pd.DataFrame({"crop": ["Rice", "Wheat"]})
    encoded_train, encoders = encode_labels(df_train, ["crop"], fit=True)
    df_test = pd.DataFrame({"crop": ["Rice", "Barley"]})
    encoded_test, _ = encode_labels(df_test, ["crop"], fit=False, encoders=encoders)
    assert encoded_train["crop"].max() >= 0
    assert encoded_test["crop"].iloc[1] == -1
