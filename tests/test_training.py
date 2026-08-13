from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.train import train_model


def create_sample_dataset():
    """Create a small synthetic dataset for testing."""

    df = pd.DataFrame(
        {
            "Pregnancies": [1, 2, 3, 4, 5, 6, 7, 8],
            "Glucose": [90, 110, 130, 150, 170, 190, 100, 120],
            "BloodPressure": [70, 72, 75, 80, 85, 90, 78, 82],
            "SkinThickness": [20, 21, 22, 23, 24, 25, 26, 27],
            "Insulin": [80, 82, 84, 90, 100, 110, 95, 105],
            "BMI": [22.5, 24.1, 26.5, 28.3, 31.4, 34.8, 29.5, 27.8],
            "DiabetesPedigreeFunction": [
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.45,
                0.55,
            ],
            "Age": [25, 30, 35, 40, 45, 50, 38, 42],
            "Outcome": [0, 0, 0, 1, 1, 1, 0, 1],
        }
    )

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )


@patch("joblib.dump")
def test_train_model_success(mock_dump):
    """
    Verify that the model trains successfully.
    """

    X_train, X_test, y_train, y_test = create_sample_dataset()

    model, accuracy, f1 = train_model(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
    assert hasattr(model, "predict")

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0

    mock_dump.assert_called_once()


@patch("joblib.dump")
def test_model_returns_predictions(mock_dump):
    """
    Verify trained model returns predictions.
    """

    X_train, X_test, y_train, y_test = create_sample_dataset()

    model, _, _ = train_model(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})


@patch("joblib.dump")
def test_prediction_probabilities(mock_dump):
    """
    Verify probability values are valid.
    """

    X_train, X_test, y_train, y_test = create_sample_dataset()

    model, _, _ = train_model(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    probabilities = model.predict_proba(X_test)

    assert probabilities.shape == (len(X_test), 2)

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


@patch("joblib.dump")
def test_accuracy_range(mock_dump):
    """
    Accuracy should lie between 0 and 1.
    """

    X_train, X_test, y_train, y_test = create_sample_dataset()

    _, accuracy, f1 = train_model(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert 0 <= accuracy <= 1
    assert 0 <= f1 <= 1


@patch("joblib.dump")
def test_training_failure_is_propagated(mock_dump):
    """Verify invalid training data raises an error without saving a model."""

    X_train, X_test, y_train, y_test = create_sample_dataset()
    invalid_y_train = pd.Series([0] * len(y_train), index=y_train.index)

    with pytest.raises(ValueError):
        train_model(
            X_train,
            invalid_y_train,
            X_test,
            y_test,
        )

    mock_dump.assert_not_called()
