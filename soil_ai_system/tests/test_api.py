"""API endpoint tests.

The ``test_predict_valid`` test requires trained model artifacts and
is skipped when the files don't exist (i.e. before Phase 2 training).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from api.main import app
from config import SAVED_MODELS_PATH

client = TestClient(app)

_MODELS_DIR = Path(__file__).resolve().parents[1] / SAVED_MODELS_PATH


def test_health():
    """Verify health endpoint returns OK status.

    Args:
        None

    Returns:
        None
    """
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


@pytest.mark.skipif(
    not (_MODELS_DIR / "xgboost_crop.pkl").exists(),
    reason="Model not trained yet — run training.train_all first",
)
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
    r = client.post("/api/v1/predict", json=payload)
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
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422
