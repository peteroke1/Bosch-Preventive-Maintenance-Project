import numpy as np
import pandas as pd


def predict_rul( machine_id: str, sensor_input: dict, model, dataset: pd.DataFrame, feature_cols: list ) -> dict:

    """
    Predict Remaining Useful Life (RUL) for a machine

    Parameters
    ----------
    machine_id : str
    sensor_input : dict
        {
            "pressure_bar": float, 
            "temp_celsius": float, 
            "flow_lpm":     float, 
            "vibration_x_g": float, 
            "vibration_y_g": float, 
            "pump_rpm":     float,        
        }
    model : trained ML model
    dataset : historical dataset (already feature-engineered base fields)
    feature_cols : list of model feature columns

    Returns 
    -------
    dict with prediction results
    """

    SENSOR_COLS = [ 
        'pressure_bar', 'temp_celsius', 'flow_lpm', 
        'vibration_x_g', 'vibration_y_g', 'pump_rpm', 'vibration_magnitude'       
    ]

    
    machine_df = dataset[
        dataset[machine_id] == machine_id]         #50
    if machine_df.empty:
        raise ValueError(f"Machine '{machine_id}' not found")

    latest = machine_df.sort_values("timestamp").iloc[-1]


    new_row = {
        "machine_id": machine_id,
        "timestamp": pd.Timestamp.now(),
        **sensor_input
    }

    # Derive feature
    new_row["vibration_magntude"] = np.sqrt(
        sensor_input["vibration_x_g"]**2 +
        sensor_input["vibration_y_g"]**2
    )

    # Pull from dataset ( DO NOT recompute)
    new_row["machine_age_days"] = latest["machine_age_days"]
    new_row["days_since_last_maintenance"] = latest["days_since_last_maintenance"]
    new_row["is_synthetic"] = latest["is_synthetic"]

    history = (
        machine_df
        .sort_values("timestamp")
        .tail(6)
    )

    combined = (
        pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    for col in SENSOR_COLS:
        combined[f"{col}_lag1"] = combined[col].shift(1)
        combined[f"{col}_lag3"] = combined[col].shift(3)
        combined[f"{col}_lag6"] = combined[col].shift(6)
        combined[f"{col}_roll6"] = combined[col].rolling(6, min_periods=1).mean()

    feature_row = combined.iloc[[-1]].copy()

    #fill missing lag values
    for col in SENSOR_COLS:
        for suf in ["_lag1", "_lag3", "_lag6", "_roll6"]:
            feat = f"{col}{suf}"
            if feature_row[feat].isna().any():
                feature_row[feat] = feature_row[col].values[0]
    
    x = feature_row[feature_cols]
    rul_hours = max(0.0, model.predict(x)[0])
    rul_days = rul_hours/24

    if rul_hours <= 24:
        status = "CRITICAL"
    elif rul_hours <= 72:
        status = "WARNING"
    elif rul_hours <= 168:
        status = "CAUTION"
    else:
        status = "NORMAL"
    
    return {
        "machine_id": machine_id,
        "rul_hours": float(rul_hours),
        "rul_days": float(rul_days),
        "status": status,
        "machine_age_days": float(new_row["machine_age_days"]),
        "days_since_last_maintenance": float(new_row["days_since_last_maintenance"]),
    }
    

                         
    
    
    
