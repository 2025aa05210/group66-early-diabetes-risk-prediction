from unittest.mock import MagicMock, patch

from api.schemas import PredictionRequest
from api.service import predict_diabetes


@patch("api.service.predictor")
@patch("api.service.preprocessor")
def test_prediction_pipeline(mock_preprocessor, mock_predictor):
    """
    End-to-end integration test:
    Request -> Preprocessing -> Model Prediction
    """

    # Mock preprocessing
    mock_preprocessor.transform.return_value = [
        [0.25] * 8
    ]

    # Mock model prediction
    mock_predictor.predict.return_value = (
        1,
        0.82,
    )

    request = PredictionRequest(
        pregnancies=2,
        glucose=120,
        blood_pressure=70,
        skin_thickness=20,
        insulin=80,
        bmi=25.5,
        diabetes_pedigree_function=0.35,
        age=30,
    )

    prediction, probability = predict_diabetes(request)

    mock_preprocessor.transform.assert_called_once()
    mock_predictor.predict.assert_called_once()

    assert prediction == 1
    assert isinstance(prediction, int)

    assert probability == 0.82
    assert isinstance(probability, float)

    assert 0 <= probability <= 1