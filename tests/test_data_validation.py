import pandas as pd
import pytest

from src.data.validator import (
    validate_dataset,
    REQUIRED_COLUMNS,
)
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


def test_valid_dataset_returns_true(valid_dataframe):
    """Verify a valid dataset returns True."""

    result = validate_dataset(valid_dataframe)

    assert result is True


def test_empty_dataset_with_required_columns_returns_true():
    """Verify an empty dataset with required columns is accepted."""

    empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    result = validate_dataset(empty_df)

    assert result is True


def test_missing_required_column_raises_exception(valid_dataframe):
    """Verify missing required column raises DataValidationError."""

    invalid_df = valid_dataframe.drop(columns=["Glucose"])

    with pytest.raises(DataValidationError):
        validate_dataset(invalid_df)


def test_non_numeric_column_is_accepted(valid_dataframe):
    """Verify non-numeric required columns are accepted."""

    valid_dataframe["Glucose"] = ["high", "low"]

    result = validate_dataset(valid_dataframe)

    assert result is True


def test_missing_value_dataset_is_accepted(valid_dataframe):
    """Verify missing values do not cause validation to fail."""

    valid_dataframe.loc[0, "BMI"] = None
    valid_dataframe.loc[1, "Age"] = None

    result = validate_dataset(valid_dataframe)

    assert result is True


def test_invalid_zero_values_are_accepted(valid_dataframe):
    """Verify zero values do not cause validation to fail."""

    valid_dataframe.loc[0, "Glucose"] = 0
    valid_dataframe.loc[1, "Insulin"] = 0
    valid_dataframe.loc[0, "BMI"] = 0

    result = validate_dataset(valid_dataframe)

    assert result is True
