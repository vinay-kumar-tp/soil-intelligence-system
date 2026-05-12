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

# Feature columns
FEATURE_COLS = [
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
    "soil_quality_index",
    "fertility_score",
    "lat_lon_cluster",
    "region_code",
    "season_encoded",
]

TARGET_CROP = "crop"
TARGET_FERTILITY = "fertility_grade"
TARGET_DEFICIENCY = "nutrient_status"

# Paths
RAW_DATA_PATH = "datasets/raw/"
PROCESSED_DATA_PATH = "datasets/processed/"
SAVED_MODELS_PATH = "saved_models/v1/"
LOG_PATH = "logs/"
REPORT_PATH = "reports/"
SHAP_OUTPUT_PATH = "reports/shap_outputs/"

# Nutrient thresholds
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
