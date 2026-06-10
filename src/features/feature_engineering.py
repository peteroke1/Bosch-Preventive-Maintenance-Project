import pandas as pd
import numpy as np
from src.data.data_cleaning import DataCleaning
from src.logger import configure_logger
import os
import json
from src.exception import MyException
from src.config.constant import feature_data_path, feature_data_json, BUCKET_NAME
from src.cloud.s3_storage import S3Storage
import io
from datetime import datetime

logging = configure_logger()

class FeatureEngineering:
    def __init__(self):
        self.data_cleaning = DataCleaning()
        self.cleaned_data = self.data_cleaning.clean_data()  
    
    def engineer_features(self):

        try:
            logging.info(f"Loaded: {len(self.cleaned_data):,} rows, {self.cleaned_data.shape[1]} cols")
            logging.info(f"Columns: {list(self.cleaned_data.columns)}")
            logging.info(f"\nRUL range: {self.cleaned_data['rul_hours'].min():.1f} - {self.cleaned_data['rul_hours'].max():.1f} hours")
            logging.info(f"Machines: {sorted(self.cleaned_data['machine_id'].unique())}")

            self.cleaned_data['installation_date'] = pd.to_datetime(self.cleaned_data['installation_date'])
            self.cleaned_data['last_filter_change_date'] = pd.to_datetime(self.cleaned_data['last_filter_change_date'])
            self.cleaned_data['timestamp'] = pd.to_datetime(self.cleaned_data['timestamp'])

            self.cleaned_data['machine_age_days'] = (
                self.cleaned_data['timestamp'] - self.cleaned_data['installation_date']
            ).dt.total_seconds() / 86400

            self.cleaned_data['is_synthetic'] = (self.cleaned_data['fluid_type'] == 'synthetic').astype(int)

            self.cleaned_data = self.cleaned_data.drop(columns=[
                'installation_date', 'last_filter_change_date', 
                'fluid_type', 'maintenance_priority'
            ])

            logging.info(f"After feature engineering: {len(self.cleaned_data):,} rows")
            logging.info(f"Columns: {list(self.cleaned_data.columns)}")
            logging.info(f"\nmachine_age_days - min: {self.cleaned_data['machine_age_days'].min():.0f},"
                         f"max: {self.cleaned_data['machine_age_days'].max():.0f}")
            logging.info(f"is_synthetic distribution:\n{self.cleaned_data['is_synthetic'].value_counts().to_string()}")

            self.cleaned_data['vibration_magnitude'] = np.sqrt(
                self.cleaned_data['vibration_x_g']**2 + self.cleaned_data['vibration_y_g']**2
            )

            logging.info("vibration magnitude compute")
            logging.info(f"min: {self.cleaned_data['vibration_magnitude'].min():.4f}")
            logging.info(f"max: {self.cleaned_data['vibration_magnitude'].max():.4f}")
            logging.info(f"mean: {self.cleaned_data['vibration_magnitude'].mean():.4f}")

            SENSOR_COLS = [
                'pressure_bar', 'temp_celsius', 'flow_lpm', 
                'vibration_x_g', 'vibration_y_g', 'pump_rpm', 'vibration_magnitude'
            ]

            self.cleaned_data = self.cleaned_data.sort_values(['machine_id', 'timestamp'])

            for col in SENSOR_COLS:
                grp = self.cleaned_data.groupby("machine_id")[col]
                self.cleaned_data[f"{col}_lag1"] = grp.shift(1)
                self.cleaned_data[f"{col}_lag3"] = grp.shift(3)
                self.cleaned_data[f"{col}_lag6"] = grp.shift(6)
                self.cleaned_data[f"{col}_roll6"] = grp.transform(lambda x: x.rolling(6, min_periods=1).mean())

            self.cleaned_data = self.cleaned_data.reset_index(drop=True)

            LAG_FEATURES = [c for c in self.cleaned_data.columns if '_lag' in c or '_roll' in c]

            print(f"After lag features: {len(self.cleaned_data):,} rows")
            print(f"Lag/roll features added: {len(LAG_FEATURES)}")
            print(f"Total columns now: {self.cleaned_data.shape[1]}")

            BASE_FEATURES_RUL = [
                'pressure_bar', 'temp_celsius', 'flow_lpm', 
                'vibration_x_g', 'vibration_y_g', 'pump_rpm', 'vibration_magnitude',
                'machine_age_days', 'days_since_last_maintenance', 'is_synthetic'
            ]
            FINAL_FEATURES_RUL = [f for f in BASE_FEATURES_RUL + LAG_FEATURES if f in self.cleaned_data.columns]

            logging.info(f"RUL features: {len(FINAL_FEATURES_RUL)}")
            rul_data = self.cleaned_data[self.cleaned_data['is_anomaly'] ==1].copy()

            logging.info(f"Shape: {rul_data.shape}")
            logging.info(f"\nMissing Values:")
            missing = rul_data[FINAL_FEATURES_RUL + ['rul_hours']].isnull().sum()
            logging.info(missing[missing > 0].to_string() if missing.any() else " None")

            logging.info(f"\nRUL distribution:")
            logging.info(rul_data['rul_hours'].describe().round(2).to_string())

            logging.info(f"\nRows per machine:")
            logging.info(rul_data.groupby('machine_id')['rul_hours'].agg(['count', 'min', 'max']).to_string())

            os.makedirs(os.path.dirname(feature_data_path), exist_ok=True)

            rul_dataset = rul_data[FINAL_FEATURES_RUL + ['rul_hours', 'machine_id', 'timestamp']]

            #Storing of the feature datasetin aws s3
            bucket_name = BUCKET_NAME
            s3 = S3Storage(bucket_name)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            csv_buffer = io.StringIO()
            rul_dataset.to_csv(csv_buffer, index=False)

            s3.upload_bytes(
                data=csv_buffer.getvalue(),
                s3_key=f"features/rul_dataset_{timestamp}.csv",
                content_type="text/csv"
            )

            json_buffer = json.dumps({
                'FINAL_FEATURES_RUL': FINAL_FEATURES_RUL,
                'SENSOR_COLS': SENSOR_COLS,                
            }, indent=2)

            s3.upload_bytes(
                data=json_buffer,
                s3_key=f"features/features_metadata_{timestamp}.json",
                content_type="application/json"
            )

            logging.info("RUL dataset and metadata uploaded directly to s3")
            logging.info(f"Rows: {len(rul_data):,}")
            logging.info(f"Features: {len(FINAL_FEATURES_RUL)}")

            return rul_data, FINAL_FEATURES_RUL
        
        except Exception as e:
            logging.error(f"Error during feature engineering: {e}")
            raise MyException(e)
        
#features = FeatureEngineering()
#features.engineer_features()

if __name__ == "__main__":
    features = FeatureEngineering()
    features.engineer_features()


