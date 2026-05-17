"""Phase 3E - FastAPI Routes.

Defines the core API endpoints for inference, explainability, and health checks.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from inference.validators import SoilInferenceRequest
from inference.engine import run_full_inference

from api.metrics import global_metrics

router = APIRouter()

@router.post("/predict", summary="Unified Soil AI Inference")
async def predict(request: SoilInferenceRequest) -> Dict[str, Any]:
    """Execute the full AI orchestration pipeline."""
    try:
        payload = request.dict()
        result = run_full_inference(payload)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail="Inference pipeline failed internally.")
            
        # Record Model Selection
        crop = result.get("predictions", {}).get("crop", {}).get("prediction", "unknown")
        global_metrics.record_prediction(crop)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during inference: {str(e)}")


@router.get("/health", summary="System Health Check")
async def health_check() -> Dict[str, str]:
    """Check if the API is active."""
    return {"status": "healthy", "service": "Soil Intelligence API Phase 5"}

@router.get("/metrics", summary="Grafana-Compatible JSON Metrics")
async def get_metrics() -> Dict[str, Any]:
    """Expose operational observability metrics."""
    return global_metrics.get_snapshot()
