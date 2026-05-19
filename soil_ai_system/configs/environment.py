import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
ARTIFACTS_DIR = PROJECT_ROOT / "saved_artifacts"
LOGS_DIR = PROJECT_ROOT / "logs" / "operations"

# Ensure directories exist
for directory in [DATASETS_DIR, SAVED_MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Environment Variables Configuration
class Config:
    ENVIRONMENT = os.getenv("APP_ENV", "production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_WORKERS = int(os.getenv("API_WORKERS", 1))
    
    # Frontend Settings
    FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 8501))
    
    # Paths
    MODEL_REGISTRY_PATH = SAVED_MODELS_DIR
    ARTIFACT_REGISTRY_PATH = ARTIFACTS_DIR
    
settings = Config()
