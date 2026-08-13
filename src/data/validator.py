"""Dataset validation and data-quality measurement.

This module validates the structure of the Pima Indians Diabetes
dataset and calculates measurable data-quality indicators required
for SEML Assignment II.
"""

import pandas as pd

from src.utils.exceptions import DataValidationError
from src.utils.logger import logger

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

# Zero is physiologically implausible and treated as invalid in these
# diagnostic measurements. Pregnancies and Outcome legitimately permit zero.
INVALID_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def validate_dataset(df: pd.DataFrame) -> dict:
    """Validate the dataset and return measurable quality results.

    The function checks:

    1. Whether the dataset is empty.
    2. Whether all required columns are present.
    3. Whether required columns contain numeric data.
    4. The number and rate of missing values.
    5. The number and rate of medically invalid zero values.

    Medically invalid zeros are reported as a data-quality issue but do
    not stop execution. This allows their extent to be measured and
    reported before an explicit imputation policy is applied.

    Args:
        df: Diabetes dataset to validate.

    Returns:
        Dictionary containing schema and data-quality measurements.

    Raises:
        DataValidationError: If the dataset is empty, has missing
            required columns, or contains non-numeric required data.
    """

    logger.info("Starting dataset validation.")

    if df is None or df.empty:
        logger.error("Dataset validation failed: dataset is empty.")
        raise DataValidationError("Dataset is empty.")

    present_columns = [column for column in REQUIRED_COLUMNS if column in df.columns]

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]

    schema_validity_percent = (len(present_columns) / len(REQUIRED_COLUMNS)) * 100

    if missing_columns:
        logger.error(
            "Dataset validation failed. Missing columns: %s",
            missing_columns,
        )
        raise DataValidationError(f"Missing required columns: {missing_columns}")

    non_numeric_columns = [
        column
        for column in REQUIRED_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        logger.error(
            "Dataset validation failed. Non-numeric columns: %s",
            non_numeric_columns,
        )
        raise DataValidationError(
            f"Non-numeric required columns: {non_numeric_columns}"
        )

    total_cells = df.shape[0] * df.shape[1]
    missing_value_count = int(df.isnull().sum().sum())

    missing_value_rate_percent = (
        missing_value_count / total_cells * 100 if total_cells else 0.0
    )

    invalid_zero_by_column = {
        column: int((df[column] == 0).sum()) for column in INVALID_ZERO_COLUMNS
    }

    invalid_zero_count = sum(invalid_zero_by_column.values())

    zero_check_cells = len(df) * len(INVALID_ZERO_COLUMNS)

    invalid_zero_rate_percent = (
        invalid_zero_count / zero_check_cells * 100 if zero_check_cells else 0.0
    )

    if missing_value_count > 0:
        logger.warning(
            "Dataset contains %s missing values (%.2f%%).",
            missing_value_count,
            missing_value_rate_percent,
        )

    if invalid_zero_count > 0:
        logger.warning(
            "Dataset contains %s medically invalid zero values "
            "(%.2f%% of checked diagnostic values).",
            invalid_zero_count,
            invalid_zero_rate_percent,
        )

    quality_results = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "schema_validity_percent": round(
            schema_validity_percent,
            2,
        ),
        "missing_value_count": missing_value_count,
        "missing_value_rate_percent": round(
            missing_value_rate_percent,
            2,
        ),
        "invalid_zero_count": invalid_zero_count,
        "invalid_zero_rate_percent": round(
            invalid_zero_rate_percent,
            2,
        ),
        "invalid_zero_by_column": invalid_zero_by_column,
    }

    logger.info(
        "Dataset validation completed successfully. "
        "Schema validity=%.2f%%, missing-value rate=%.2f%%, "
        "invalid-zero rate=%.2f%%.",
        quality_results["schema_validity_percent"],
        quality_results["missing_value_rate_percent"],
        quality_results["invalid_zero_rate_percent"],
    )

    return quality_results
