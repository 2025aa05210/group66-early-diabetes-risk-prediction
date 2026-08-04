from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(
    y_true,
    y_pred,
):

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
        ),
        "F1 Score": f1_score(
            y_true,
            y_pred,
        ),
    }
