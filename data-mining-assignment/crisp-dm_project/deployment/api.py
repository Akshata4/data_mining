from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class PredictionRequest(BaseModel):
    feature1: float  # Add more features as needed

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    data = [[request.feature1]]  # Adapt for multiple input features
    prediction = load_model_and_predict(data)
    return {"prediction": prediction}

def load_model_and_predict(data):
    model = joblib.load('deployment/model_v1.joblib')
    transformer = joblib.load('deployment/transformer.joblib')
    processed_data = transformer.transform(data)
    return model.predict(processed_data)
