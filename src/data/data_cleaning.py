import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from src.logger import configure_logger
from src.exception import MyException

logging = configure_logger()

import os
from src.config.constant import cleaned_data as DATA_DIR
from src.data.data_processing import DataProcessing

class DataCleaning:
    def __init__(self):
        self.processor = DataProcessing()
        self.maintenance_data = self.processor.data_merging()
        
    def clean_data(self):
        try:
            df = self.maintenance_data
            logging.info(f"Missing Values Before Cleaning:\n{df.isnull().sum()}")
            df = df.sort_values(by=['machine_id', 'timestamp'])

            sensor_cols = ['pressure_bar', 'temp_celsius', 'flow_lpm', 'vibration_x_g', 'vibration_y_g', 'pump_rpm',]

            df[sensor_cols] = df.groupby("machine_id")[sensor_cols].transform(
                lambda group: group.interpolate(method="linear")
            )

            df[sensor_cols] = df[sensor_cols].fillna("No Failure")

            logging.info(f"Keeping anomalies rows for RUL modeling")
            logging.info(f"Rows: {len(df):,}")
            logging.info(f"Machines: {sorted(df['machine_id'].unique())}")
            logging.info(f"RUL range: {df['rul_hours'].min():.1f} - {df['rul_hours'].max():.1f} hours")
            os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)
            df.to_csv(DATA_DIR, index=False)
            logging.info(f"Cleaned data save to {DATA_DIR}")
            return df
        except Exception as e:
            logging.error(f"Error during data cleaning: {e}")
            raise MyException(e)