import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn",
    version="1.0"
)

# Paths from .env
MODEL_PATH = os.getenv("MODEL_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")
FEATURES_PATH = os.getenv("FEATURES_PATH")

# Global variables for model artifacts
model = None
scaler = None
features = None

# Input schema
class CustomerData(BaseModel):
    data: dict

# Load model artifacts on startup
@app.on_event("startup")
def load_model():
    global model, scaler, features
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print("Error loading model artifacts:", e)

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(customer: CustomerData):
    global model, scaler, features

    if model is None or scaler is None or features is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([customer.data])

        # Ensure feature order matches training
        df = df.reindex(columns=features, fill_value=0)

        # Scale features
        scaled_data = scaler.transform(df)

        # Predict
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")