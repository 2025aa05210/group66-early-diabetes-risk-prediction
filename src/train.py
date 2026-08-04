import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.config import MODEL_PATH
from src.utils.logger import logger


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
):

    logger.info("Training Logistic Regression model.")

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    logger.info(
        f"Training completed. Accuracy={accuracy:.4f}"
    )

    joblib.dump(model, MODEL_PATH)

    logger.info("Model saved successfully.")

    return model, accuracy
