import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
equipment_data = os.path.join(BASE_DIR, "dataset", "equipment_master.csv")
failure_data = os.path.join(BASE_DIR, "dataset", "failure_labels.csv")
maintenance_data = os.path.join(BASE_DIR, "dataset", "maintenance_log.csv")
sensor_data = os.path.join(BASE_DIR, "dataset", "sensor_telemetry_cleaneed.csv")

cleaned_data = os.path.join(BASE_DIR, "dataset", "cleaned_data.csv", "bosch_enriched_data.csv")
feature_data_path = os.path.join(BASE_DIR, "dataset", "final_data", "rul_featured.csv")
feature_data_json = os.path.join(BASE_DIR, "dataset","final_data", "rul_features.json")

BUCKET_NAME = "grp-feature-engineered-bucket"     #"bosch-predictive-maintenance"
S3_ARTIFACT_ROOT = f"s3://{BUCKET_NAME}/mlflow.artifacts"
DAGSHUB_OWNER = "peteroke1" 
DAGSHUB_REPO = "Predictive-Maintenance-for-Hydrolic-system-Bosch-Rexroth"
EXPERIMENT = "bosch_rul_prediction"

