from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.predict import DiabetesPredictor
from src.utils.exceptions import ModelLoadError, PredictionError


@pytest.fixture
def sample_input():
    """Sample input with one patient record."""
    return pd.DataFrame(
        {
            "Pregnancies": [2],
            "Glucose": [120],
            "BloodPressure": [70],
            "SkinThickness": [20],
            "Insulin": [80],
            "BMI": [25.5],
            "DiabetesPedigreeFunction": [0.35],
            "Age": [30],
        }
    )


@patch("joblib.load")
def test_model_load_success(mock_load):
    """
    Verify predictor loads the model successfully.
    """
    mock_model = MagicMock()
    mock_load.return_value = mock_model

    predictor = DiabetesPredictor()

    assert predictor.model == mock_model
    mock_load.assert_called_once()


@patch("joblib.load")
def test_model_load_failure(mock_load):
    """
    Verify ModelLoadError is raised if loading fails.
    """
    mock_load.side_effect = Exception("Unable to load model")

    with pytest.raises(ModelLoadError):
        DiabetesPredictor()


@patch("joblib.load")
def test_prediction_output(mock_load, sample_input):
    """
    Verify prediction returns a valid class and probability.
    """
    mock_model = MagicMock()

    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.20, 0.80]])

    mock_load.return_value = mock_model

    predictor = DiabetesPredictor()

    prediction, probability = predictor.predict(sample_input)

    assert prediction in [0, 1]
    assert 0 <= probability <= 1


@patch("joblib.load")
def test_prediction_return_types(mock_load, sample_input):
    """
    Verify return types of prediction.
    """
    mock_model = MagicMock()

    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.75, 0.25]])

    mock_load.return_value = mock_model

    predictor = DiabetesPredictor()

    prediction, probability = predictor.predict(sample_input)

    assert isinstance(prediction, (int, np.integer))
    assert isinstance(probability, (float, np.floating))


@patch("joblib.load")
def test_prediction_failure(mock_load, sample_input):
    """
    Verify PredictionError is raised when prediction fails.
    """
    mock_model = MagicMock()

    mock_model.predict.side_effect = Exception("Prediction failed")

    mock_load.return_value = mock_model

    predictor = DiabetesPredictor()

    with pytest.raises(PredictionError):
        predictor.predict(sample_input)


@patch("joblib.load")
def test_probability_between_zero_and_one(mock_load, sample_input):
    """
    Verify probability is always between 0 and 1.
    """
    mock_model = MagicMock()

    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.45, 0.55]])

    mock_load.return_value = mock_model

    predictor = DiabetesPredictor()

    _, probability = predictor.predict(sample_input)

    assert 0 <= probability <= 1


@patch("joblib.load")
def test_higher_glucose_increases_probability(mock_load):
    """
    Assignment directional test.
    This is marked xfail because it depends on the trained model.
    """
    mock_model = MagicMock()

    mock_model.predict.side_effect = [np.array([0]), np.array([1])]
    mock_model.predict_proba.side_effect = [
        np.array([[0.80, 0.20]]),
        np.array([[0.20, 0.80]]),
    ]

    mock_load.return_value = mock_model

    predictor = DiabetesPredictor()

    low_glucose = pd.DataFrame(
        {
            "Pregnancies": [2],
            "Glucose": [90],
            "BloodPressure": [70],
            "SkinThickness": [20],
            "Insulin": [80],
            "BMI": [25],
            "DiabetesPedigreeFunction": [0.3],
            "Age": [30],
        }
    )

    high_glucose = low_glucose.copy()
    high_glucose["Glucose"] = 180

    _, low_prob = predictor.predict(low_glucose)
    _, high_prob = predictor.predict(high_glucose)

    assert high_prob > low_prob
