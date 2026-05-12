# AI-Powered Spatial Soil Intelligence System

## Project Overview
Multi-task learning and ensemble system for crop suitability, fertility grading, nutrient deficiency detection, and actionable recommendations with explainability.

## Architecture
Ingest and validate data, engineer features, train baseline and advanced models, assemble ensembles, and expose inference through API and dashboard.

## Installation
Create and activate a virtual environment, then install dependencies from requirements.txt.

## Folder Structure
Core modules live under preprocessing, models, training, inference, explainability, api, dashboard, and tests. Datasets and artifacts are under datasets and saved_models.

## Execution Pipeline
Prepare datasets, run training pipeline, validate tests, then launch API and dashboard for inference.

## Tech Stack
Python, TensorFlow/Keras, XGBoost, SHAP, Streamlit, FastAPI, MLflow, SQLite.

## Model Architecture
Baseline classical models, XGBoost classifiers, multi-head DNN, and stacked ensemble with SHAP explainability and contrastive reasoning.

## Future Scope
Add richer spatial data, refine seasonal rules, automate data pipelines, and productionize monitoring.
