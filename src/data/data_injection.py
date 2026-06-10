import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
from src.config.constant import equipment_data, failure_data, sensor_data, maintenance_data, cleaned_data
from src.logger import configure_logger
from src.exception import MyException

logging = configure_logger()

class DataIngestion:
    def __init__(self):
        self.equipment_data = equipment_data
        self.failure_data = failure_data
        self.sensor_data = sensor_data
        self.maintenance_data = maintenance_data
    
    def ingest_data(self):
        try:
            equipment_df = pd.read_csv(self.equipment_data)
            failure_df = pd.read_csv(self.failure_data)
            sensor_df = pd.read_csv(self.sensor_data)
            maintenance_df = pd.read_csv(self.maintenance_data)
            print(failure_df.head())
            print(equipment_df.head())
            print(sensor_df.head())
            print(maintenance_df.head())
            logging.info("Data ingested successfully.")

            return equipment_df, failure_df, sensor_df,  maintenance_df
        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise MyException(e)







