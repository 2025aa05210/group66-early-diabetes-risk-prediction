from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"

# Models
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Logs
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"
