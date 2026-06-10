import os
import tempfile
import joblib
import boto3
import mlflow
import dagshub
from src.logger import configure_logger
from src.connections.mlflow_setup import setup_mlflow
import mlflow.sklearn
from src.cloud.s3_storage import S3Storage
from src.config.constant import BUCKET_NAME
logging = configure_logger()


class ModelRegistry:                            #37

    def __init__(self):
        self.bucket_name = BUCKET_NAME
        setup_mlflow()
        self._upload_to_s3 = S3Storage(bucket_name=self.bucket_name)._upload_to_s3
    
    def register(self, model, model_name: str, params: dict, metrics: dict, tags: dict=None) -> dict:
        tags = tags or {}

        with mlflow.start_run(tags={"model_name": model_name, **tags}) as run:
            run_id = run.info.run_id

            # 1. Log  hyperparameters
            mlflow.log_params(params)
            logging.info(f"Params logged: {params}")

            # 2. Log metrics
            mlflow.log_metrics({
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            })
            logging.info(
                f"Metrics logged: RMSE: {metrics['rmse']:.2f} | "
                f"MAE: {metrics['mae']:.2f} | R2: {metrics['r2']:.3f}"
            )

            # 3. Log + register in ONE call - no separate register_model() needed
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name,   #handle registration internally                
            )
            logging.info(f"Model logged and resgistered as '{model_name}'")

            # 4. Backup .joblib to your S3 bucket
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, f"{model_name}.joblib")
                joblib.dump(model, local_path)
                s3_uri= self._upload_to_s3(local_path, model_name, run_id)
            
            summary = {
                "run_id":       run_id,
                "s3_uri":       s3_uri,
                "model_name":   model_name,
                "metrics":      metrics,
            }
            logging.info("Model registeration completed.")
            return summary
        
        


