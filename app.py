from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("sales_forecast_model.pkl")

app = FastAPI()

class SalesPredictionRequest(BaseModel):
    store_id: int
    product_id: int
    past_sales: list

@app.get("/")
def home():
    return {"message": "AI Sales Forecasting API is running!"}

@app.post("/predict")
def predict_sales(request: SalesPredictionRequest):
    features = np.array(request.past_sales).reshape(1, -1)
    forecast = model.predict(features)[0]
    return {"store_id": request.store_id, "product_id": request.product_id, "forecast": forecast}
