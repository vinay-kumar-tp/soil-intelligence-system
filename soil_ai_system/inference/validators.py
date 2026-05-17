"""Phase 3A - Validation and Schemas.

Validates inference payloads securely without contaminating the main pipeline.
Provides Pydantic schemas for the incoming FastAPI requests.
"""

from typing import Optional

from pydantic import BaseModel, Field, root_validator


class SoilInferenceRequest(BaseModel):
    """Structured input schema for the unified soil AI inference."""
    
    # Core NPK and continuous features
    N: float = Field(..., description="Nitrogen content ratio (e.g., kg/ha or raw ratio)")
    P: float = Field(..., description="Phosphorus content ratio")
    K: float = Field(..., description="Potassium content ratio")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Relative Humidity percentage (0-100)")
    ph: float = Field(..., description="Soil pH value (1-14)")
    rainfall: float = Field(..., description="Rainfall in mm")
    
    # Optional context features
    region: Optional[str] = Field(None, description="Optional region name for localized recommendations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "N": 90,
                "P": 42,
                "K": 43,
                "temperature": 20.8,
                "humidity": 82.0,
                "ph": 6.5,
                "rainfall": 202.9,
                "region": "northern_plains"
            }
        }
