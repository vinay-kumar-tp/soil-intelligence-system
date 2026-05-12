import random
import numpy as np
import tensorflow as tf

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
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

# Raw datasets
RAW_DATASETS = {
    "crop_recommendation": "crop_recommendation.csv",
    "soil_fertility": "soil_fertility.csv",
    "optional_india_soil_data": "optional_india_soil_data.csv",
}

# Feature columns
RAW_FEATURE_COLS = [
    "N",
    "P",
    "K",
    "ph",
    "ec",
    "organic_carbon",
    "moisture",
    "temperature",
    "humidity",
    "rainfall",
    "latitude",
    "longitude",
    "season",
]

ENGINEERED_FEATURE_COLS = [
    "soil_quality_index",
    "fertility_score",
    "soil_health_score",
    "lat_lon_cluster",
    "region_code",
    "season_encoded",
]

FEATURE_COLS = RAW_FEATURE_COLS + ENGINEERED_FEATURE_COLS

TARGET_CROP = "crop"
TARGET_FERTILITY = "fertility_grade"
TARGET_DEFICIENCY = "nutrient_status"

TARGET_COLS = [TARGET_CROP, TARGET_FERTILITY, TARGET_DEFICIENCY]
COLUMN_RENAME_MAP = {}
LABEL_NORMALIZATION_COLS = TARGET_COLS

# Paths
RAW_DATA_PATH = "datasets/raw/"
PROCESSED_DATA_PATH = "datasets/processed/"
SAVED_MODELS_PATH = "saved_models/v1/"
LOG_PATH = "logs/"
REPORT_PATH = "reports/"
SHAP_OUTPUT_PATH = "reports/shap_outputs/"

PROCESSED_MERGED_FILENAME = "merged_soil_data.csv"
RAW_DATA_REPORT_FILENAME = "raw_data_report.txt"
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
