PROJECT CONTEXT FOR ANTIGRAVITY AI AGENT
========================================
Project: Soil Intelligence System
Phase: Phase 1B (Completed) -> Phase 2 (Modeling) READY

ARCHITECTURE OVERVIEW:
The system uses a Modular Dataset-Specific Pipeline architecture.
Instead of merging datasets with incompatible schemas, it processes each dataset independently
using shared logic with specialized configurations.

CORE COMPONENTS:
- soil_ai_system/config.py: Centralized config for features, targets, rename maps, physical bounds, and feature column lists.
- soil_ai_system/preprocessing/validator.py: Severity-based validation (HARD violations vs SOFT warnings).
- soil_ai_system/preprocessing/cleaner.py: IQR-based outlier CLIPPING (prevents data loss).
- soil_ai_system/preprocessing/feature_engineer.py: Feature engineering + encode_season() + derive_nutrient_status().
- soil_ai_system/preprocessing/feature_store.py: Resilient artifact load/save (handles missing files gracefully).
- soil_ai_system/preprocessing/pipeline_runner.py: Main entry point for processing all datasets.
- soil_ai_system/saved_artifacts/{dataset}/: Versioned label encoders and scalers.

DATASETS PROCESSED:
- crop (2200 rows): Target is crop (10 classes). Features: N,P,K,temperature,humidity,ph,rainfall,fertility_score
- fertility (880 rows): Target is fertility_grade (3 classes). Features: N,P,K,ph,ec,organic_carbon,S,Zn,Fe,Cu,Mn,B,fertility_score,soil_quality_index
- regional (875 rows): No target. Features: N,P,K,ph,region_code

TRAINING ARCHITECTURE (Phase 2):
- training/train_all.py: Master orchestrator — loads processed CSVs per-dataset, trains per-task.
- training/train_crop.py: Baselines + XGBoost + Ensemble for crop classification.
- training/train_fertility.py: XGBoost for fertility grading.
- training/train_deficiency.py: XGBoost for derived nutrient deficiency.
- DNN multi-head model trains crop/fertility/deficiency simultaneously.

INFERENCE PIPELINE:
- inference/preprocess_input.py: Validates raw input, engineers features, scales with saved scaler.
- inference/predict.py: Lazy model loading, XGBoost+DNN predictions, SHAP explainability, recommendations.
- api/predictor.py: Lazy bridge between FastAPI and inference layer.
- All model loading is deferred to first inference call (no import-time crashes).

STATE OF CODING:
All Phase 1B preprocessing logic implemented and verified.
Phase 2 training code fully implemented, ready to execute.
All cross-module bugs fixed (FEATURE_COLS, encode_season, merged_soil_data, etc.).