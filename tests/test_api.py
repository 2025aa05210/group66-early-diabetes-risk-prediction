from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def valid_request():
    return {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 25.5,
        "DiabetesPedigreeFunction": 0.35,
        "Age": 30,
    }


def test_root_endpoint():
    """Verify the API information endpoint returns HTTP 200."""

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Early Diabetes Risk Prediction API",
        "status": "running",
    }


def test_health_endpoint():
    """
    Verify /health returns HTTP 200.
    """

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@patch("api.main.predict_diabetes")
def test_predict_valid_request(mock_predict):
    """
    Verify a valid prediction request succeeds.
    """

    mock_predict.return_value = (1, 0.87)

    response = client.post(
        "/predict",
        json=valid_request(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] == 1
    assert body["probability"] == 0.87


def test_predict_missing_field():
    """
    Verify missing required field returns HTTP 422.
    """

    request = valid_request()

    request.pop("Glucose")

    response = client.post(
        "/predict",
        json=request,
    )

    assert response.status_code == 422


def test_predict_invalid_type():
    """
    Verify invalid field type returns HTTP 422.
    """

    request = valid_request()

    request["Age"] = "Thirty"

    response = client.post(
        "/predict",
        json=request,
    )

    assert response.status_code == 422


@patch("api.main.predict_diabetes")
def test_predict_internal_server_error(mock_predict):
    """
    Verify unexpected errors return HTTP 500
    without exposing stack traces.
    """

    mock_predict.side_effect = Exception("Unexpected failure")

    response = client.post(
        "/predict",
        json=valid_request(),
    )

    assert response.status_code == 500

    body = response.json()

    assert body["detail"] == "Internal server error"
