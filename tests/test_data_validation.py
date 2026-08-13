import pandas as pd
import pytest

from src.data.validator import REQUIRED_COLUMNS, validate_dataset
from src.utils.exceptions import DataValidationError


@pytest.fixture
def valid_dataframe():
    """Create a valid sample dataframe."""
    return pd.DataFrame(
        {
            "Pregnancies": [2, 4],
            "Glucose": [120, 150],
            "BloodPressure": [70, 80],
            "SkinThickness": [20, 25],
            "Insulin": [80, 90],
            "BMI": [25.5, 30.2],
            "DiabetesPedigreeFunction": [0.35, 0.67],
            "Age": [30, 45],
            "Outcome": [0, 1],
        }
    )


def test_valid_dataset_returns_quality_metrics(valid_dataframe):
    """Verify a valid dataset returns the expected quality metrics."""

    result = validate_dataset(valid_dataframe)

    assert isinstance(result, dict)
    assert result["row_count"] == 2
    assert result["column_count"] == len(REQUIRED_COLUMNS)
    assert result["schema_validity_percent"] == 100.0
    assert result["missing_value_count"] == 0
    assert result["missing_value_rate_percent"] == 0.0
    assert result["invalid_zero_count"] == 0
    assert result["invalid_zero_rate_percent"] == 0.0


def test_empty_dataset_raises_exception():
    """Verify an empty dataset raises DataValidationError."""

    empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    with pytest.raises(DataValidationError, match="Dataset is empty"):
        validate_dataset(empty_df)


def test_missing_required_column_raises_exception(valid_dataframe):
    """Verify missing required column raises DataValidationError."""

    invalid_df = valid_dataframe.drop(columns=["Glucose"])

    with pytest.raises(DataValidationError):
        validate_dataset(invalid_df)


def test_non_numeric_column_raises_exception(valid_dataframe):
    """Verify non-numeric required columns raise DataValidationError."""

    valid_dataframe["Glucose"] = ["high", "low"]

    with pytest.raises(DataValidationError, match="Non-numeric"):
        validate_dataset(valid_dataframe)


def test_missing_value_metrics_are_calculated(valid_dataframe):
    """Verify missing-value count and rate are calculated correctly."""

    valid_dataframe.loc[0, "BMI"] = None
    valid_dataframe.loc[1, "Age"] = None

    result = validate_dataset(valid_dataframe)

    assert result["missing_value_count"] == 2
    assert result["missing_value_rate_percent"] == 11.11


def test_invalid_zero_metrics_are_calculated(valid_dataframe):
    """Verify invalid-zero count, rate, and per-column values."""

    valid_dataframe.loc[0, "Glucose"] = 0
    valid_dataframe.loc[1, "Insulin"] = 0
    valid_dataframe.loc[0, "BMI"] = 0

    result = validate_dataset(valid_dataframe)

    assert result["invalid_zero_count"] == 3
    assert result["invalid_zero_rate_percent"] == 30.0
    assert result["invalid_zero_by_column"] == {
        "Glucose": 1,
        "BloodPressure": 0,
        "SkinThickness": 0,
        "Insulin": 1,
        "BMI": 1,
    }
