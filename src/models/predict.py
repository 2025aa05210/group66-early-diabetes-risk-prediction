import joblib

from src.config import MODEL_PATH
from src.utils.logger import logger
from src.utils.exceptions import (
    ModelLoadError,
    PredictionError,
)


class DiabetesPredictor:

    def __init__(self):

        try:

            logger.info("Loading prediction model.")

            self.model = joblib.load(MODEL_PATH)

            logger.info("Model loaded successfully.")

        except Exception as e:
            logger.error(e)
            raise ModelLoadError(str(e))

    def predict(self, data):

        try:

            prediction = self.model.predict(data)[0]

            probability = self.model.predict_proba(data)[0][1]

            logger.info(
                f"Prediction completed. Probability={probability:.4f}"
            )

            return prediction, probability

        except Exception as e:

            logger.error(e)

            raise PredictionError(str(e))
