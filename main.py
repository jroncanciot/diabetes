import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Clase de entrada
class ApiInput(BaseModel):
  features: List[float]

# Clase de salida
class ApiOutput(BaseModel):
  forecast: int

# Creación del API
app = FastAPI()
model = joblib.load("model.joblib")

# Endpoint del API tipo post
@app.post("/diabetes_prediction")
async def diabetes_prediction(data: ApiInput) -> ApiOutput:
  predict = model.predict([data.features]).flatten().tolist()
  pred = ApiOutput(forecast = predict[0])
  return pred
