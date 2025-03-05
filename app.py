from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np

# Load the trained XGBoost model
model = xgb.Booster()
model.load_model("sales_forecast_model.json")

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
    # Convert input data into XGBoost DMatrix format
    features = np.array(request.past_sales).reshape(1, -1)
    dmatrix = xgb.DMatrix(features)

    # Make prediction
    forecast = model.predict(dmatrix)[0]  # Ensure correct format

    return {
        "store_id": request.store_id,
        "product_id": request.product_id,
        "forecast": float(forecast)  # Convert to standard float for JSON response
    }
