import pandas as pd

from src.config import DATA_PATH
from src.utils.exceptions import DataValidationError
from src.utils.logger import logger


def load_data(path=DATA_PATH):
    """
    Load diabetes dataset.
    """

    try:
        logger.info(f"Loading dataset from {path}")

        df = pd.read_csv(path)

        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

        return df

    except FileNotFoundError as e:
        logger.error("Dataset file not found.")
        raise DataValidationError(str(e))

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
