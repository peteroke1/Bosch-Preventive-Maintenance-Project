import boto3
import pandas as pd
import numpy as np
import io
import json
from src.logger import configure_logger

logging = configure_logger()

class S3Storage:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.s3 = boto3.client("s3")
    
    def upload_bytes(self, data, s3_key, content_type):
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=data,
            ContentType=content_type
        )
    
    def load_csv(self, csv_key):
        try:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=csv_key)

            data = pd.read_csv(
                io.BytesIO(obj["Body"].read()),
                parse_dates=["timestamp"]
            )

            logging.info(f"Loaded CSV from S3: {csv_key} | Rows: {len(data):,}")
            return data
        
        except Exception as e:
            logging.error(f"Error loading CSV from S3: {e}")
            raise

    def load_json(self, json_key):
        try:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=json_key)

            features = json.loads(obj["Body"].read().decode("utf-8"))

            logging.info(f"Loaded JSON from S3: {json_key}")
            return features
        
        except Exception as e:
            logging.error(f"Error loading JSON from S3: {e}")
            raise
    
    def get_latest_file(self, prefix, keyword):
        try:
            response = self.s3.list_objects_v2(
                Bucket= self.bucket_name,
                Prefix=prefix
            )

            if "Contents" not in response:
                raise ValueError("No file found in S3 prefix")
            
            # Filter only relevant files
            files = [
                obj for obj in response["Contents"]
                if keyword in obj["Key"]
            ]

            if not files:
                raise ValueError(f"No files found with keyword: {keyword}")
            
            #pick latest by LastModified
            latest_file = max(files, key=lambda x: x["LastModified"])

            logging.info(f"Lastest file selected: {latest_file['Key']}")

            return latest_file["Key"]
        
        except Exception as e:
            logging.error(f"Error finding latest file: {e}")
            raise

    def _upload_to_s3(self, local_path: str, model_name: str, run_id: str) -> str:
        """Upload model file to s3 and return the full S3 URI."""
        s3_key = f"mlflow-artifacts/{run_id}/{model_name}.joblib"
        boto3.client("s3").upload_file(local_path, self.bucket_name, s3_key)
        s3_uri = f"s3://{self.bucket_name}/{s3_key}"
        logging.info(f"Model uploaded to {s3_uri}")
        return s3_uri