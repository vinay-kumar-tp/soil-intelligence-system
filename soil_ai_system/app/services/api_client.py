import requests
import logging
import time
import os
from typing import Dict, Any, Optional

logger = logging.getLogger("frontend.api_client")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
if not API_BASE_URL.endswith("/api/v1"):
    API_BASE_URL = f"{API_BASE_URL.rstrip('/')}/api/v1"
    
TIMEOUT_SECONDS = 10

# Create a robust session that ignores system proxies (resolves WinError 10061)
session = requests.Session()
session.trust_env = False

def get_system_health() -> Dict[str, Any]:
    """Check if the AI backend is active."""
    try:
        start = time.time()
        response = session.get(f"{API_BASE_URL}/health", timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        latency = (time.time() - start) * 1000
        data["latency_ms"] = round(latency, 2)
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unreachable", "error": str(e), "latency_ms": 0}

def get_system_metrics() -> Dict[str, Any]:
    """Fetch operational observability metrics from the backend."""
    try:
        response = session.get(f"{API_BASE_URL}/metrics", timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def predict_soil(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send a prediction payload to the unified engine."""
    try:
        response = session.post(f"{API_BASE_URL}/predict", json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Prediction request failed: {e}")
        return {"status": "error", "message": str(e)}
