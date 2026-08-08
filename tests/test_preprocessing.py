from unittest.mock import MagicMock, patch

import pandas as pd

from src.features.preprocessing import DataPreprocessor


@patch("src.features.preprocessing.joblib.load")
def test_data_preprocessor_loads_scaler(mock_load):
    """Verify the preprocessor loads the scaler successfully."""
    mock_scaler = MagicMock()
    mock_load.return_value = mock_scaler

    preprocessor = DataPreprocessor()

    assert preprocessor.scaler is mock_scaler
    mock_load.assert_called_once()


@patch("src.features.preprocessing.joblib.load")
def test_data_preprocessor_transform_calls_scaler_transform(mock_load):
    """Verify transform calls the scaler transform method."""
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = [[1.0, 2.0, 3.0]]
    mock_load.return_value = mock_scaler

    preprocessor = DataPreprocessor()
    data = pd.DataFrame(
        {
            "Pregnancies": [1],
            "Glucose": [100],
            "BloodPressure": [70],
            "SkinThickness": [20],
            "Insulin": [80],
            "BMI": [25.5],
            "DiabetesPedigreeFunction": [0.3],
            "Age": [30],
        }
    )

    result = preprocessor.transform(data)

    mock_scaler.transform.assert_called_once_with(data)
    assert result == [[1.0, 2.0, 3.0]]
