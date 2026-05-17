"""Phase 3A - Model and Artifact Loaders.

Handles lazy loading and caching of inference models and preprocessing artifacts.
Ensures models are only loaded once per worker and prevents disk bottlenecks.
Provides graceful fallbacks if models are missing.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from api.metrics import global_metrics

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Optional TensorFlow import (only loaded if a Keras model is requested)
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger("inference.loaders")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ModelRegistryCache:
    """Thread-safe lazy loader for models and preprocessing artifacts."""

    _instance: Optional[ModelRegistryCache] = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelRegistryCache:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelRegistryCache, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self) -> None:
        """Initialize the empty cache."""
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, Any] = {}
        self._registry_data: Optional[list] = None
        
        self.registry_path = _PROJECT_ROOT / "model_registry.json"
        
        # We assume Phase 1 artifacts are here
        self.artifacts_dir = _PROJECT_ROOT / "soil_ai_system" / "saved_artifacts"
        
        # Lock for dictionary mutations
        self._cache_lock = threading.Lock()

    def _load_registry(self) -> list:
        """Load the registry JSON from disk if not loaded."""
        if self._registry_data is not None:
            return self._registry_data
            
        if not self.registry_path.exists():
            # Try finding it inside soil_ai_system if the root differs
            alt_path = _PROJECT_ROOT / "soil_ai_system" / "model_registry.json"
            if alt_path.exists():
                self.registry_path = alt_path
            else:
                logger.error("model_registry.json not found at %s", self.registry_path)
                return []
                
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self._registry_data = json.load(f)
            logger.info("Loaded model registry with %d entries", len(self._registry_data))
        except Exception as e:
            logger.error("Failed to parse registry: %s", e)
            self._registry_data = []
            
        return self._registry_data

    def get_best_model_info(self, task: str, model_name: str = "XGBoost") -> Optional[Dict[str, Any]]:
        """Retrieve metadata for the requested model for a task.
        
        Args:
            task: Task name (e.g., 'crop', 'fertility', 'deficiency', 'crop+fertility+deficiency')
            model_name: Optional explicit model name to load (defaults to XGBoost)
            
        Returns:
            Registry entry dict or None.
        """
        registry = self._load_registry()
        task_models = [m for m in registry if m.get("task") == task and m.get("artifact_exists")]
        
        if not task_models:
            logger.warning("No valid models found in registry for task: %s", task)
            return None
            
        # Try to find the exact model_name first
        target = [m for m in task_models if m.get("model_name") == model_name]
        if target:
            return target[0]
            
        # Select best based on primary accuracy metric as fallback
        def _get_score(m: dict) -> float:
            metrics = m.get("metrics", {})
            return metrics.get("test_accuracy", metrics.get("test_crop_accuracy", 0.0))
            
        best = max(task_models, key=_get_score)
        return best

    def load_model(self, task: str) -> Optional[Any]:
        """Lazy load the best model for a given task.
        
        Args:
            task: Task name (e.g., 'crop')
            
        Returns:
            Fitted model artifact (joblib object or Keras model).
        """
        with self._cache_lock:
            if task in self._models:
                logger.info("Cache hit for task '%s'. Returning model.", task)
                global_metrics.record_cache_hit()
                return self._models[task]

        logger.info("Cache miss for task '%s'. Locating model...", task)
        global_metrics.record_cache_miss()
        best_info = self.get_best_model_info(task)
        if not best_info:
            return None
            
        artifact_path = Path(best_info["artifact_path"])
        if not artifact_path.exists():
            logger.error("Artifact missing at path: %s", artifact_path)
            return None

        # Load the artifact
        try:
            if artifact_path.suffix == ".keras":
                if not TF_AVAILABLE:
                    logger.error("TensorFlow is required but not installed.")
                    return None
                logger.info("Loading Keras model from %s", artifact_path)
                model = keras.models.load_model(str(artifact_path))
            else:
                logger.info("Loading Joblib model from %s", artifact_path)
                model = joblib.load(artifact_path)
                
            with self._cache_lock:
                self._models[task] = model
            logger.info("Successfully loaded and cached model for '%s'", task)
            return model
            
        except Exception as e:
            logger.error("Failed to load model %s: %s", artifact_path, e)
            return None

    def load_scaler(self, scaler_name: str) -> Optional[Any]:
        """Lazy load a scaler artifact.
        
        Args:
            scaler_name: Filename of the scaler (e.g., 'crop_scaler.pkl').
            
        Returns:
            Fitted sklearn scaler or None.
        """
        with self._cache_lock:
            if scaler_name in self._scalers:
                logger.info("Cache hit for scaler '%s'. Returning scaler.", scaler_name)
                global_metrics.record_cache_hit()
                return self._scalers[scaler_name]
        
        logger.info("Cache miss for scaler '%s'. Locating...", scaler_name)
        global_metrics.record_cache_miss()
                
        scaler_path = self.artifacts_dir / scaler_name
        if not scaler_path.exists():
            logger.error("Scaler artifact missing: %s", scaler_path)
            return None
            
        try:
            logger.info("Loading scaler from %s", scaler_path)
            scaler = joblib.load(scaler_path)
            with self._cache_lock:
                self._scalers[scaler_name] = scaler
            return scaler
        except Exception as e:
            logger.error("Failed to load scaler %s: %s", scaler_path, e)
            return None

    def load_label_encoder(self, encoder_name: str) -> Optional[Any]:
        """Lazy load a label encoder artifact.
        
        Args:
            encoder_name: Filename of the encoder (e.g., 'crop_label_encoder.pkl').
            
        Returns:
            Fitted sklearn LabelEncoder or None.
        """
        # Shares the same dictionary as scalers for simplicity
        return self.load_scaler(encoder_name)


# Global singleton access point
registry_cache = ModelRegistryCache()
