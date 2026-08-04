import numpy as np

from src.features.preprocessing import DataPreprocessor
from src.models.predict import DiabetesPredictor

preprocessor = DataPreprocessor()
predictor = DiabetesPredictor()


def predict_diabetes(request):

    input_data = np.array([[
        request.pregnancies,
        request.glucose,
        request.blood_pressure,
        request.skin_thickness,
        request.insulin,
        request.bmi,
        request.diabetes_pedigree,
        request.age
    ]])

    scaled = preprocessor.transform(input_data)

    prediction, probability = predictor.predict(scaled)

    result = "High Risk" if prediction == 1 else "Low Risk"

    return result, float(probability)
