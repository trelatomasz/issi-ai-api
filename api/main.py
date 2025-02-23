from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Wczytanie modelu
from model.california_housing import load_model

model = load_model()

# Definicja struktury danych wejściowych
class HousingFeatures(BaseModel):
    MedInc: Optional[float] = Field(default=None, description="Median income in block group")
    HouseAge: Optional[float] = Field(default=None, description="Median house age in block group")
    AveRooms: Optional[float] = Field(default=None, description="Average number of rooms per household")
    AveBedrms: Optional[float] = Field(default=None, description="Average number of bedrooms per household")
    Population: Optional[float] = Field(default=None, description="Total population in block group")
    AveOccup: Optional[float] = Field(default=None, description="Average number of household members")
    Latitude: Optional[float] = Field(default=None, description="Latitude of block group")
    Longitude: Optional[float] = Field(default=None, description="Longitude of block group")

# Definicja struktury odpowiedzi
class PredictionResponse(BaseModel):
    predicted_house_value: float = Field(description="Predicted median house value for the input features")
app = FastAPI()
# Endpoint do predykcji
@app.post("/predict")
def predict(features: HousingFeatures) -> PredictionResponse:
    try:
        # Konwersja danych wejściowych na tablicę numpy
        input_data = np.array([
            features.MedInc,
            features.HouseAge,
            features.AveRooms,
            features.AveBedrms,
            features.Population,
            features.AveOccup,
            features.Latitude,
            features.Longitude
        ]).reshape(1, -1)

        # Wykonanie predykcji
        prediction = model.predict(input_data)
        print(prediction)
        return PredictionResponse(predicted_house_value=float(prediction[0]))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
