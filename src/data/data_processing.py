import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from src.logger import configure_logger
from src.exception import MyException
from src.data.data_injection import DataIngestion

logging = configure_logger()

class DataProcessing:

    def __init__(self):
            data_ingestion = DataIngestion()
            self.equipment_data, self.failure_data, self.sensor_data, self.maintenance_data = data_ingestion.ingest_data()

    def data_merging(self):
        try:
            self.equipment_data['installation_date'] = pd.to_datetime(self.equipment_data['installation_date'])
            self.equipment_data['last_filter_change_date'] = pd.to_datetime(self.equipment_data['last_filter_change_date'])
            self.failure_data['failure_timestamp'] = pd.to_datetime(self.failure_data['failure_timestamp'])
            self.failure_data['degradation_start_timestamp'] = pd.to_datetime(self.failure_data['degradation_start_timestamp'])
            self.sensor_data['timestamp'] = pd.to_datetime(self.sensor_data['timestamp'])
            self.maintenance_data['action_timestamp'] = pd.to_datetime(self.maintenance_data['action_timestamp'])

            logging.info("Data loaded successfully")
            logging.info(f"sensor_data: {len(self.sensor_data):,} rows")
            logging.info(f"equipment_data: {len(self.equipment_data):,} rows")
            logging.info(f"failure_data: {len(self.failure_data):,} rows")
            logging.info(f"maintenance-data: {len(self.maintenance_data):,} rows")

            self.sensor_data['ts_num'] = self.sensor_data['timestamp'].view('int64') // 10**9
            self.maintenance_data['ts_num'] = self.maintenance_data['action_timestamp'].view('int64') // 10**9

            merged_data = pd.merge_asof(
                 self.sensor_data.sort_values('ts_num'),
                 self.maintenance_data.sort_values('ts_num')[['machine_id', 'ts_num']].rename(columns={'ts_num': 'maint_ts'}),
                 left_on='ts_num', right_on='maint_ts',
                 by='machine_id', direction='backward'
            )

            merged_data['days_since_last_maintenance'] = ((merged_data['ts_num'] - merged_data['maint_ts']) / 86400).fillna(90)
            merged_data = merged_data.drop(columns=['ts_num', 'maint_ts'])
            merged_data = merged_data.sort_values(['machine_id', 'timestamp']).reset_index(drop=True)

            logging.info(f"After maintenance merge: {len(merged_data):,} rows")
            logging.info(f"days_since_last_maintenance min: {merged_data['days_since_last_maintenance'].min():.1f},"
                f"max:{merged_data['days_since_last_maintenance'].max():.1f},"
                f"nulls: {merged_data['days_since_last_maintenance'].isna().sum()}")
            
            merged_data = merged_data.merge(self.equipment_data, on='machine_id', how='left')

            logging.info(f"After equipment merge: {len(merged_data):,} rows, {merged_data.shape[1]} cols")
            logging.info(f"Columns: {list(merged_data.columns)}")
            logging.info(merged_data.head())
            logging.info(merged_data.info())

            return merged_data
        except Exception as e:
             logging.error(f"Error during data merging: {e}")
             raise MyException(e)



