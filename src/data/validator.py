import pandas as pd

from src.utils.logger import logger
from src.utils.exceptions import DataValidationError

REQUIRED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def validate_dataset(df: pd.DataFrame):

    logger.info("Validating dataset.")

    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        raise DataValidationError(
            f"Missing columns: {missing_columns}"
        )

    if df.isnull().sum().sum() > 0:
        logger.warning("Dataset contains missing values.")

    logger.info("Dataset validation successful.")

    return True
