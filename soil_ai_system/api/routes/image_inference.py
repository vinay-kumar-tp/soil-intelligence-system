"""Phase 3E - FastAPI Routes for Image Inference.

Endpoint to handle soil image uploads and return CNN predictions.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Dict, Any

from inference.soil_image_predictor import predict_from_image
from api.metrics import global_metrics
from utils.logger import get_logger

logger = get_logger("api_image", "api.log")
router = APIRouter()


@router.post("/predict-from-image", summary="Soil Image Classification")
async def predict_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Classify soil type from an uploaded image using CNN and recommend crops."""
    logger.info(f"Received image upload: {file.filename} (content_type={file.content_type})")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    try:
        contents = await file.read()
        
        if not contents:
             raise HTTPException(status_code=400, detail="Empty image file provided.")
             
        result = predict_from_image(contents)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "Inference failed."))
            
        # Record Model Selection/Usage
        global_metrics.record_prediction(result.get("soil_type", "unknown"))
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
