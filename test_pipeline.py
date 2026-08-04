from src.data.loader import load_data
from src.data.validator import validate_dataset
from src.features.preprocessing import DataPreprocessor
from src.models.predict import DiabetesPredictor

# Load and validate dataset
df = load_data()
validate_dataset(df)

# Select one sample (exclude Outcome column)
sample = df.drop(columns=["Outcome"]).head(1)

# Preprocess
preprocessor = DataPreprocessor()
sample_scaled = preprocessor.transform(sample)

# Predict
predictor = DiabetesPredictor()
prediction, probability = predictor.predict(sample_scaled)

print(f"Prediction: {prediction}")
print(f"Probability: {probability:.4f}")
