from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20)
    glucose: float = Field(..., ge=0)
    blood_pressure: float = Field(..., ge=0)
    skin_thickness: float = Field(..., ge=0)
    insulin: float = Field(..., ge=0)
    bmi: float = Field(..., ge=0)
    diabetes_pedigree: float = Field(..., ge=0)
    age: int = Field(..., ge=1, le=120)


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
