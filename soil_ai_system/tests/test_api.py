from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    """Verify health endpoint returns OK status.

    Args:
        None

    Returns:
        None
    """
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_predict_valid():
    """Verify predict endpoint accepts valid payloads.

    Args:
        None

    Returns:
        None
    """
    payload = {
        "N": 50,
        "P": 40,
        "K": 40,
        "ph": 6.5,
        "moisture": 45,
        "humidity": 65,
        "rainfall": 800,
        "temperature": 28,
        "ec": 0.5,
        "organic_carbon": 0.8,
        "state": "Tamil Nadu",
        "season": "kharif",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "crop" in data
    assert "soil_health_score" in data
    assert 0 <= data["confidence_crop"] <= 1


def test_predict_invalid_ph():
    """Verify predict endpoint rejects invalid pH values.

    Args:
        None

    Returns:
        None
    """
    payload = {
        "N": 50,
        "P": 40,
        "K": 40,
        "ph": 20,
        "moisture": 45,
        "humidity": 65,
        "rainfall": 800,
        "temperature": 28,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
