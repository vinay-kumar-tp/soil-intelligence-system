"""Phase 3A - Validation and Schemas.

Validates inference payloads securely without contaminating the main pipeline.
Provides Pydantic schemas for the incoming FastAPI requests.
"""

from typing import Optional, Dict, Any

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
    
    # Phase 6X: Spatial Agronomic Intelligence fields
    region_zone: Optional[str] = Field(None, description="Region Zone (e.g., southern_india)")
    state: Optional[str] = Field(None, description="State (e.g., Tamil Nadu)")
    district: Optional[str] = Field(None, description="District (e.g., Chennai)")
    taluk: Optional[str] = Field(None, description="Taluk / Tehsil")
    hobli: Optional[str] = Field(None, description="Hobli / Block")
    village: Optional[str] = Field(None, description="Village")
    agro_climatic_zone: Optional[str] = Field(None, description="Agro-Climatic Zone (e.g., Coastal)")
    irrigation_type: Optional[str] = Field(None, description="Irrigation Type (e.g., Canal irrigation)")
    soil_texture: Optional[str] = Field(None, description="Soil Texture (e.g., Clay)")
    seasonal_context: Optional[str] = Field(None, description="Seasonal Context (e.g., Kharif)")
    # Phase 6Z: Real-time Weather Intelligence fields
    latitude: Optional[float] = Field(None, description="Latitude of target coordinates")
    longitude: Optional[float] = Field(None, description="Longitude of target coordinates")
    weather_context: Optional[Dict[str, Any]] = Field(None, description="Cached or pre-fetched live weather report")
    
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
