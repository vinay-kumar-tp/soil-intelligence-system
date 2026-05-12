from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class SoilInput(BaseModel):
    N: float = Field(..., ge=0, le=200, description="Nitrogen kg/ha")
    P: float = Field(..., ge=0, le=200, description="Phosphorus kg/ha")
    K: float = Field(..., ge=0, le=200, description="Potassium kg/ha")
    ph: float = Field(..., ge=0, le=14)
    ec: float = Field(default=0.5, ge=0, le=5)
    organic_carbon: float = Field(default=0.5, ge=0, le=5)
    moisture: float = Field(..., ge=0, le=100)
    temperature: float = Field(..., ge=-10, le=60)
    humidity: float = Field(..., ge=0, le=100)
    rainfall: float = Field(..., ge=0, le=5000)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    state: Optional[str] = None
    season: Optional[str] = None


class PredictionResponse(BaseModel):
    crop: str
    confidence_crop: float
    fertility_grade: str
    confidence_fertility: float
    nutrient_status: str
    fertilizer_recommendations: List[str]
    irrigation_suggestion: str
    seasonal_advice: str
    crop_action_guide: str
    shap_top_features: List[Dict]
    contrastive_explanation: Dict
    soil_health_score: float
