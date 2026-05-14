import random
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if tf is not None:
    tf.random.set_seed(SEED)

# Class labels
CROP_LABELS = [
    "Rice",
    "Wheat",
    "Sugarcane",
    "Groundnut",
    "Cotton",
    "Banana",
    "Maize",
    "Coconut",
    "Jowar",
    "Ragi",
]
FERTILITY_LABELS = ["Low", "Medium", "High"]
DEFICIENCY_LABELS = [
    "Nitrogen deficient",
    "Phosphorus deficient",
    "Potassium deficient",
    "Balanced",
]
IRRIGATION_LABELS = ["Irrigation needed", "Optimal moisture", "Overwatered"]
SEASON_LABELS = ["kharif", "rabi", "summer"]

# Raw datasets (dataset-specific pipelines)
CROP_DATASET_KEY = "crop"
FERTILITY_DATASET_KEY = "fertility"
REGIONAL_DATASET_KEY = "regional"

RAW_DATASETS = {
    CROP_DATASET_KEY: "Crop_recommendation.csv",
    FERTILITY_DATASET_KEY: "soil_fertility.csv",
    REGIONAL_DATASET_KEY: "southern_india_soil.csv",
}

# Feature groups
CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_TARGET = "crop"

MICRONUTRIENT_FEATURES = ["S", "Zn", "Fe", "Cu", "Mn", "B"]
FERTILITY_FEATURES = ["N", "P", "K", "ph", "ec", "organic_carbon"] + MICRONUTRIENT_FEATURES
FERTILITY_TARGET = "fertility_grade"

REGIONAL_FEATURES = ["N", "P", "K", "ph"]
REGIONAL_TARGET = None

OPTIONAL_FEATURES = []

# Engineered features (dataset-specific)
CROP_ENGINEERED_FEATURES = ["fertility_score"]
CROP_OPTIONAL_ENGINEERED_FEATURES = ["soil_quality_index"]
FERTILITY_ENGINEERED_FEATURES = ["fertility_score", "soil_quality_index"]
REGIONAL_ENGINEERED_FEATURES = ["region_code"]

ENGINEERED_FEATURE_COLS = list(
    dict.fromkeys(
        CROP_ENGINEERED_FEATURES
        + CROP_OPTIONAL_ENGINEERED_FEATURES
        + FERTILITY_ENGINEERED_FEATURES
        + REGIONAL_ENGINEERED_FEATURES
    )
)

TARGET_DEFICIENCY = "nutrient_status"
TARGET_COLS = [CROP_TARGET, FERTILITY_TARGET]

COLUMN_RENAME_MAP = {
    "label": "crop",
    "Output": "fertility_grade",
    "pH": "ph",
    "EC": "ec",
    "OC": "organic_carbon",
    "Nitrogen Value": "N",
    "Phosphorous value": "P",
    "Potassium value": "K",
    "District": "region",
}

LABEL_NORMALIZATION_COLS = [CROP_TARGET, FERTILITY_TARGET, "region"]

PIPELINE_CONFIGS = {
    CROP_DATASET_KEY: {
        "features": CROP_FEATURES,
        "target": CROP_TARGET,
        "engineered_features": CROP_ENGINEERED_FEATURES,
        "optional_engineered_features": CROP_OPTIONAL_ENGINEERED_FEATURES,
    },
    FERTILITY_DATASET_KEY: {
        "features": FERTILITY_FEATURES,
        "target": FERTILITY_TARGET,
        "engineered_features": FERTILITY_ENGINEERED_FEATURES,
        "optional_engineered_features": [],
    },
    REGIONAL_DATASET_KEY: {
        "features": REGIONAL_FEATURES,
        "target": REGIONAL_TARGET,
        "engineered_features": REGIONAL_ENGINEERED_FEATURES,
        "optional_engineered_features": [],
    },
}

# Paths
RAW_DATA_PATH = "datasets/raw/"
PROCESSED_DATA_PATH = "datasets/processed/"
SAVED_MODELS_PATH = "saved_models/v1/"
ARTIFACTS_PATH = "saved_artifacts/"
PIPELINE_ARTIFACTS = {
    CROP_DATASET_KEY: "saved_artifacts/crop_pipeline",
    FERTILITY_DATASET_KEY: "saved_artifacts/fertility_pipeline",
    REGIONAL_DATASET_KEY: "saved_artifacts/regional_pipeline",
}
LOG_PATH = "logs/"
REPORT_PATH = "reports/"
SHAP_OUTPUT_PATH = "reports/shap_outputs/"

PROCESSED_DATASETS = {
    CROP_DATASET_KEY: "crop_processed.csv",
    FERTILITY_DATASET_KEY: "fertility_processed.csv",
    REGIONAL_DATASET_KEY: "regional_processed.csv",
}
RAW_DATA_REPORT_FILENAME = "raw_data_report.txt"
SCHEMA_RECONCILIATION_REPORT = "schema_reconciliation_report.txt"
PREPROCESSING_PIPELINE_REPORT = "preprocessing_pipeline_report.txt"
PHASE1B_FINAL_AUDIT_REPORT = "phase1b_final_audit.txt"
PROCESSED_DATA_REPORT_FILENAME = "processed_data_summary.txt"
PREPROCESSING_LOG_FILE = "preprocessing.log"
LABEL_ENCODERS_FILENAME = "label_encoders.pkl"
SCALER_FILENAME = "scaler.pkl"
KMEANS_FILENAME = "kmeans_spatial.pkl"
UNKNOWN_LABEL_VALUE = -1

# Nutrient thresholds
N_MIN = 0
N_MAX = 200
P_MIN = 0
P_MAX = 200
K_MIN = 0
K_MAX = 200
PH_MIN_ALLOWED = 0.0
PH_MAX_ALLOWED = 14.0
MOISTURE_MIN = 0.0
MOISTURE_MAX = 100.0
HUMIDITY_MIN = 0.0
HUMIDITY_MAX = 100.0
RAINFALL_MIN = 0.0
RAINFALL_MAX = 5000.0
TEMPERATURE_MIN = -10.0
TEMPERATURE_MAX = 60.0
N_LOW = 20
N_HIGH = 80
P_LOW = 10
P_HIGH = 60
K_LOW = 15
K_HIGH = 80
MOISTURE_LOW = 25
MOISTURE_HIGH = 75
PH_MIN = 5.5
PH_MAX = 7.5
EC_MAX = 2.0

# Train/val/test split
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# DNN hyperparameters
DNN_EPOCHS = 100
DNN_BATCH_SIZE = 32
DNN_LR = 0.001
DNN_PATIENCE = 10

# XGBoost defaults
XGB_MAX_DEPTH = 6
XGB_LR = 0.1
XGB_N_ESTIMATORS = 300

# Spatial clustering
SPATIAL_CLUSTERS = 10

# Model version
MODEL_VERSION = "v1"
