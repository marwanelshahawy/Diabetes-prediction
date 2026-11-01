from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Paths
RAW_DATA = ROOT / "data" / "raw" / "diabetes_prediction_dataset.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"

# Cols
NUM_COLS = ["age", "bmi", "blood_glucose_level", "HbA1c_level"]
CAT_COLS = ["gender", "smoking_history"]
BIN_COLS = ["hypertension", "heart_disease", "bmi_missing"]
TARGET = "diabetes"

# Values
BMI_IMPUTED_VALUE = 27.32
THRESHOLD = 0.505
RANDOM_STATE = 42

# best params
XGB_PARAMS = {
    "n_estimators": 170,
    "learning_rate": 0.11784576811599486,
    "max_depth": 3,
    "min_child_weight": 8,
    "subsample": 0.8560652947304449,
    "colsample_bytree": 0.6595461607901565,
    "reg_lambda": 2.451501525033984,
    "scale_pos_weight": 8.245308107972834,
}