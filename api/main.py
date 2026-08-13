from fastapi import FastAPI, HTTPException

from api.schemas import PredictionRequest, PredictionResponse
from api.service import predict_diabetes
from src.utils.logger import logger

app = FastAPI(
    title="Early Diabetes Risk Prediction API",
    description="SEML Assignment II - Group 66",
    version="1.0.0",
)


@app.get("/")
def root():

    logger.info("Root endpoint accessed.")

    return {"message": "Early Diabetes Risk Prediction API", "status": "running"}


@app.get("/health")
def health():

    logger.info("Health endpoint accessed.")

    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        prediction, probability = predict_diabetes(request)

        logger.info(f"Prediction successful: {prediction}")

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
        )

    except Exception:
        logger.exception("Prediction failed.")

        raise HTTPException(status_code=500, detail="Internal server error")
