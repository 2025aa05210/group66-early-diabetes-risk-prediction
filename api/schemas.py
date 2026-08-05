from pydantic import BaseModel, Field, ConfigDict


class PredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pregnancies: int = Field(alias="Pregnancies", ge=0, le=20)
    glucose: float = Field(alias="Glucose", gt=0, le=300)
    blood_pressure: float = Field(alias="BloodPressure", gt=0, le=200)
    skin_thickness: float = Field(alias="SkinThickness", ge=0, le=100)
    insulin: float = Field(alias="Insulin", ge=0, le=1000)
    bmi: float = Field(alias="BMI", gt=0, le=80)
    diabetes_pedigree_function: float = Field(
        alias="DiabetesPedigreeFunction",
        ge=0,
        le=5
    )
    age: int = Field(alias="Age", ge=1, le=120)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float = Field(ge=0.0, le=1.0)