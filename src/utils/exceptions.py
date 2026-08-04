class DiabetesPredictionException(Exception):
    """Base exception for the project."""

    pass


class DataValidationError(DiabetesPredictionException):
    """Raised when dataset validation fails."""

    pass


class ModelLoadError(DiabetesPredictionException):
    """Raised when model cannot be loaded."""

    pass


class PredictionError(DiabetesPredictionException):
    """Raised during prediction."""

    pass
