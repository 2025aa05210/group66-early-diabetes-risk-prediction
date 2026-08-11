import joblib
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # ------------------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------------------
    data_path = Path("data") / "diabetes.csv"

    df = pd.read_csv(data_path)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # ------------------------------------------------------------------
    # Train-Test Split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ------------------------------------------------------------------
    # Feature Scaling
    # ------------------------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # Train Model
    # ------------------------------------------------------------------
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )

    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # Save Model
    # ------------------------------------------------------------------
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    joblib.dump(model, model_dir / "diabetes_model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")

    print("\nModel regenerated successfully.")
    print("Scaler regenerated successfully.")
    print(f"Saved to: {model_dir.resolve()}")


if __name__ == "__main__":
    main()