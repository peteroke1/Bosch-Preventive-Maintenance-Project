import numpy as np
import pandas as pd
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from src.cloud.s3_storage import S3Storage
from src.features.feature_engineering import FeatureEngineering
from src.logger import configure_logger
from src.registry.model_registry import ModelRegistry
from src.config.constant import BUCKET_NAME
logging = configure_logger()


class ModelingPipeline:
    def __init__(self, bucket_name=BUCKET_NAME):
        self.s3 = S3Storage(bucket_name)

        self.rul_data = None
        self.feature_sets = None
        self.FINAL_FEATURES_RUL = None
        self.SENSOR_COLS = None

        self.model = None
    

    def run_feature_engineering(self):
        logging.info("Starting feature engineering...")
        FeatureEngineering().engineer_features()
        logging.info("feature engineering complete. DAta uploaded to s3.")
    

    def load_data(self):
        csv_key = self.s3.get_latest_file(
            prefix="feature/",
            keyword="rul_dataset"
        )

        json_key = self.s3.get_latest_file(
            prefix="features/",
            keyword="features_metadata"
        )

        self.rul_data = self.s3.load_csv(csv_key)
        self.feature_sets = self.s3.load_json(json_key)

        self.FINAL_FEATURES_RUL = self.feature_sets["FINAL_FEATURES_RUL"]
        self.SENSOR_COLS = self.feature_sets["SENSOR_COLS"]

        logging.info(f"Loaded data: {len(self.rul_data):,} rows")
    

    def prepare_data(self):
        self.rul_data = self.rul_data.sort_values("timestamp")

        split_time = self.rul_data["timestamp"].quantile(0.8)

        train = self.rul_data[self.rul_data["timestamp"] <= split_time]
        test = self.rul_data[self.rul_data["timestamp"] > split_time]

        train = train.dropna(subset=self.FINAL_FEATURES_RUL + ["rul_hours"])
        test = test.dropna(subset=self.FINAL_FEATURES_RUL + ["rul_hours"])

        self.x_train = train[self.FINAL_FEATURES_RUL]
        self.y_train = train["rul_hours"]

        self.x_test = test[self.FINAL_FEATURES_RUL]
        self.y_test = test["rul_hours"]

        logging.info(f"Train: {len(self.x_train):,} rows | Test: {len(self.x_test):,} rows")
    

    def train_model(self):
        logging.info("Training Model...")
        self.model = RandomForestRegressor(            
            n_estimators=300, 
            max_depth=12, 
            min_samples_leaf=10, 
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(self.x_train, self.y_train)
        logging.info("Model training complete.")

    def evaluate(self):
        preds = self.model.predict(self.x_test)


        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        mae = mean_absolute_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)

        logging.info("\nModel Performance: ")
        logging.info(f" RMSE: {rmse:.2f} ")
        logging.info(f" MAE: {mae:.2f} ")
        logging.info(f" R²: {r2:.3f}")

        return rmse, mae, r2, self.model        #35
    

    def run(self):
        self.run_feature_engineering()
        self.load_data()
        self.prepare_data()
        self.train_model()

        rmse, mae, r2, self.model = self.evaluate()

        #Hand off to registry
        registry = ModelRegistry()

        registry.register(
            model=self.model,
            model_name= "RandomForest_RUL",
            params={
               "n_estimators":       300, 
                "max_depth":          12, 
                "min_samples_leaf":   10, 
                "random_state":       42,
            },
            metrics={"rmse": rmse, "mae": mae, "r2": r2},
            tags={"stage": "dev", "dataset": "hydralic_system"},
        )


        