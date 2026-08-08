import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.config import MODEL_PATH
from src.utils.logger import logger


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Train a Logistic Regression model and save it.

    Returns:
        model: Trained model
        accuracy: Test accuracy
        f1: Test F1 score
    """

    try:
        logger.info("Training Logistic Regression model.")

        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        logger.info(f"Training completed successfully.")
        logger.info(f"Accuracy : {accuracy:.4f}")
        logger.info(f"F1 Score : {f1:.4f}")

        joblib.dump(model, MODEL_PATH)

        logger.info(f"Model saved successfully at {MODEL_PATH}")

        return model, accuracy, f1

    except Exception:
        logger.exception("Model training failed.")
        raise
