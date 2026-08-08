from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.utils.logger import logger


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance.

    Returns:
        dict: Evaluation metrics
    """

    try:
        metrics = {
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred), 4),
            "Recall": round(recall_score(y_true, y_pred), 4),
            "F1 Score": round(f1_score(y_true, y_pred), 4),
        }

        logger.info("Model evaluation completed successfully.")
        logger.info(f"Accuracy : {metrics['Accuracy']}")
        logger.info(f"Precision: {metrics['Precision']}")
        logger.info(f"Recall   : {metrics['Recall']}")
        logger.info(f"F1 Score : {metrics['F1 Score']}")

        return metrics

    except Exception:
        logger.exception("Model evaluation failed.")
        raise
