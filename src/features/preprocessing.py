import joblib

from src.config import SCALER_PATH
from src.utils.logger import logger


class DataPreprocessor:

    def __init__(self):
        logger.info("Loading scaler.")

        self.scaler = joblib.load(SCALER_PATH)

    def transform(self, data):

        logger.info("Scaling input data.")

        return self.scaler.transform(data)
