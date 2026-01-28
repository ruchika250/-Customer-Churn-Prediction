from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

app = FastAPI(
    title="ML Model API",
    description="FastAPI for chunk prediction",
)

class InputData(BaseModel):
    Age: float
    Gender: int
    Tenure: float
    Usage_Frequency: float
    Support_Calls: float
    Payment_Delays: float 
    Subscription_Type: int
    Contract_Length: int
    Total_Spend: float
    Last_Interaction: float

@app.post("/predict")
def predict(data: InputData):
    
    input_array = np.array([[ 
        data.Age,
        data.Gender,
        data.Tenure,
        data.Usage_Frequency,
        data.Support_Calls,
        data.Payment_Delays,
        data.Subscription_Type,
        data.Contract_Length,
        data.Total_Spend,
        data.Last_Interaction
    ]])

    prediction = model.predict(input_array)

    return {
        "prediction": float(prediction[0])
    }

@app.get("/")
def home():
    return {"message": "FastAPI ML Model is running."}
    predicted = prediction[0]
    