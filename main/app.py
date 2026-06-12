from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
from io import BytesIO
import boto3
import os
import joblib
import json


from src.connections.mlflow_setup import setup_mlflow
from src.cloud.s3_storage import S3Storage
from src.config.constant import BUCKET_NAME
from src.pipeline.prediction import predict_rul
from src.pipeline.training import TrainingPipeline
from src.logger import configure_logger
from src.exception import MyException


logging = configure_logger()

app = FastAPI(title="RUL Prediction API")

setup_mlflow()


MODEL_NAME = "RandomForest_RUL"
MODEL_STAGE = "latest"

s3 = S3Storage(BUCKET_NAME)


def load_artifacts():
    try:
        """Load model, dataset, and feature columns"""
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

        json_key = s3.get_latest_file(prefix="features/", keyword="features_metadata")
        feature_cols = s3.load_json(json_key)["FINAL_FEATURES_RUL"]

        csv_key = s3.get_latest_file(prefix="features/", keyword="rul_dataset")
        dataset = s3.load_csv(csv_key)
        dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
        logging.info(f"Artifacts loaded: model '{MODEL_NAME}' and dataset with {len(dataset):,} rows")

        return model, dataset, feature_cols
    except Exception as e:
        logging.error(f"Error occurred while loading artifacts: {e}")
        raise


# Load at startup
model, dataset, feature_cols = load_artifacts()

# REQUEST SCHEMA

class SensorInput(BaseModel):
    machine_id: str
    pressure_bar: float 
    temp_celsius: float
    flow_lpm: float
    vibration_x_g: float
    vibration_y_g: float
    pump_rpm: float


# ROUTES

@app.get("/")
def home():
    return {"message": "RUL Prediction API is running"}

# Training route
@app.post("/train")
def train_model():
    global model, dataset, feature_cols

    try:
        pipeline = TrainingPipeline()
        pipeline.train()

        # Reload updated artifacts
        model, dataset, feature_cols = load_artifacts()

        return {
            "status": "Model training completed and artifacts reloaded"
        }
    except Exception as e:
        logging.error(f"Error during training {e}")
        raise MyException(f"Training failed: {e}")
    

# Prediction route
@app.post("/predict")
def predict(data: SensorInput):

    sensor_input = {
        "pressure_bar": data.pressure_bar, 
        "temp_celsius": data.temp_celsius, 
        "flow_lpm":     data.flow_lpm, 
        "vibration_x_g": data.vibration_x_g, 
        "vibration_y_g": data.vibration_y_g, 
        "pump_rpm":     data.pump_rpm,  
        
    }

    try:
        result = predict_rul(
            machine_id=data.machine_id,
            sensor_input=sensor_input,
            model=model,
            dataset=dataset,

        )

        return result
    
    except ValueError as e:
        #Known issues
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        #Unexpected errors
        raise MyException(f"Prediction failed: {e}")



