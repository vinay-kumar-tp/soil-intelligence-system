# Phase 3A: Inference Architecture Refactor
- `inference/loaders.py`: Thread-safe, lazy-loading model cache based on `model_registry.json`.
- `inference/preprocessors.py`: Routing for scaling and encoding per task pipeline (crop vs fertility).
- `inference/validators.py`: Input schema validation (basic dictionary checks, delegating complex validation to Pydantic later).

# Phase 3B: Unified Prediction Engine
- `inference/predictors.py`: Model wrappers for XGBoost/DNN inference.
- `inference/engine.py`: The `run_full_inference()` pipeline coordinating loaders, preprocessors, and predictors.

# Phase 3C: Recommendation Intelligence Layer
- `inference/recommenders.py`: Rule-guided explainable recommendations based on prediction outputs.

# Phase 3D: Explainability Serving
- `inference/explainers.py`: SHAP generation and contrastive explainability logic tailored for inference inputs.

# Phase 3E: FastAPI Production Server
- `api/` structure (routes, schemas, services, middleware, dependencies).

# Phase 3F: Robustness & Failure Handling
- Fallback mechanics, comprehensive error logging in `logs/inference/`.
- Tests for all of the above.
