import mlflow
import dagshub
import joblib
import boto3
import os
import tempfile
from src.config.constant import DAGSHUB_OWNER, DAGSHUB_REPO, EXPERIMENT
from dotenv import load_dotenv
load_dotenv(override=True)


def setup_mlflow():
   dagshub.init(
       repo_owner=DAGSHUB_OWNER,
       repo_name=DAGSHUB_REPO,
       mlflow=True
   )
   mlflow.set_experiment(EXPERIMENT)

# experiment_name = EXPERIMENT
# def setup_mlflow():
#     dagshub_token = os.getenv("MLFLOW_TOKEN")
#     if not dagshub_token:
#         raise EnvironmentError("MLFLOW_TOKEN environment variable is not set")
    
#     os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
#     os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

#     dagshub_url = "http://dagshub.com"
#     repo_owner = DAGSHUB_OWNER
#     repo_name = DAGSHUB_REPO
#     #Set up mlflow tracking URL
#     mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
#     mlflow.set_experiment(experiment_name)




