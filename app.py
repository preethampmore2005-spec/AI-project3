from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(
    title="Mall Customer Segmentation API"
)

# Load model
model = joblib.load("model.pkl")

class Customer(BaseModel):
    gender: int
    age: int
    annual_income: int
    spending_score: int

@app.get("/")
def home():
    return {
        "message": "Mall Customer API Running"
    }

@app.post("/predict")
def predict(customer: Customer):

    features = np.array([
        customer.gender,
        customer.age,
        customer.annual_income,
        customer.spending_score
    ]).reshape(1, -1)

    prediction = model.predict(features)

    return {
        "segment": int(prediction[0]),
        "segment_name":
            "High Value Customer"
            if prediction[0] == 1
            else "Low Value Customer"
    }